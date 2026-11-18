# -*- coding: utf-8 -*-
"""
光谱指数增强模型训练 (E2/E3, 阶段二新增)
=========================================
python train_indices.py --mode ms  --quick          # E2 快速验证 (2 epoch)
python train_indices.py --mode ms  --epochs 80 --batch_size 4 --lr 0.001   # E2 正式
python train_indices.py --mode rgb --epochs 80 --batch_size 4 --lr 0.001   # E3 正式
python train_indices.py --mode ms  --no-resume      # 忽略断点从头训练

两种模式:
  --mode ms   (E2): 10ch 输入 [b1..b5, pan, ndi x4] + EdgePrior(Sobel pan/b2) -> 12ch conv1
              科学问题: 指数通道是否在原始波段之上有额外增益? (对照: train.py 0.6214)
  --mode rgb  (E3): 7ch 输入 [b1,b2,b3, ndi x4] + EdgePrior(RGB 灰度 Sobel) -> 8ch conv1
              科学问题: RGB+5 指数能否逼近完整多光谱? (对照: train_rgb.py 0.6097)

关键设计 (与 HANDOFF 坑 #17 同类, 勿改):
  - 指数通道在几何/辐射增强完成后从波段 on-the-fly 重算:
    ±15° 旋转的双线性插值不保持 (a-b)/(a+b), 预计算指数再增强会污染数值.
  - E2 复用 SegmentationAugmentor (6ch) 原有增强链 → 与 train.py 完全一致;
    E3 用 _augment_5band (gamma 对 5 波段反射率物理正确).
  - 指数通道 conv1 零初始化 → t=0 与对照模型函数一致, 增益归因干净.

其余 (v4 损失/epochs 80/patience 20/Warmup+EMA/梯度裁剪/NaN 保险丝/balance
score/滑窗验证) 通过复用 train_rgb.py 的实现保证逐行一致.

输出: outputs/train_indices_ms/ 或 outputs/train_indices_rgb/
"""
import json
import sys
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (MASKS_DIR, SPLIT_PATH, OUTPUT_DIR, CONFIGS_DIR,
                    CLASS_NAMES, NUM_CLASSES, H, W, IGNORE_INDEX, CLASS_PIXEL_COUNTS)
from build_input_tensor import (load_6channel, load_5band, load_indices_input,
                                compute_index_channels, normalize_tensor,
                                load_norm_stats_indices, INDEX_NAMES)
from augment import SegmentationAugmentor, compute_sample_weights
from model import DeepLabV3PlusIndices, DeepLabV3PlusRGBIndices
from losses import CombinedLoss, compute_focal_alpha
from predict_utils import sliding_window_predict
# 复用 train_rgb 的训练机制 (WarmupCosine/EMA/MetricTracker/train_one_epoch/
# balance score/曲线与混淆矩阵绘图) — 保证与已有 5 组超参逐行一致
from train_rgb import (WarmupCosineScheduler, EMA, MetricTracker,
                        compute_balance_score, train_one_epoch,
                        plot_confusion, plot_curves)

DEFECT_CLASSES = ["crack", "biological_attachment", "flaking"]
MODES = {
    "ms":  {"dir": "train_indices_ms",  "stats": "norm_stats_indices.json",
            "n_in": 10, "model": "DeepLabV3PlusIndices"},
    "rgb": {"dir": "train_indices_rgb", "stats": "norm_stats_rgb_indices.json",
            "n_in": 7, "model": "DeepLabV3PlusRGBIndices"},
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IndexDataset(Dataset):
    """E2/E3 数据集: 波段增强完成后重算指数, 再拼接归一化."""

    def __init__(self, split_data, norm_stats, augmentor=None, mode="ms"):
        self.samples = split_data
        self.norm_stats = norm_stats
        self.augmentor = augmentor
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mask_path = MASKS_DIR / s["category"] / f"{s['stem']}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)
        if self.augmentor:
            if self.mode == "ms":
                # E2: 标准 6ch 增强链 (与 train.py 完全一致), 增强后重算指数
                base, mask = self.augmentor(load_6channel(s["stem"], s["category"]), mask)
                idx = compute_index_channels(base[0:5])
                image = normalize_tensor(np.concatenate([base, idx], axis=0),
                                         self.norm_stats)          # (10, H, W)
            else:
                # E3: 5 波段联合增强 (几何一致 + gamma 物理正确), 增强后重算指数
                bands5, mask = self._augment_5band(load_5band(s["stem"], s["category"]), mask)
                idx = compute_index_channels(bands5)
                image = normalize_tensor(np.concatenate([bands5[0:3], idx], axis=0),
                                         self.norm_stats)          # (7, H, W)
        else:
            image = normalize_tensor(load_indices_input(s["stem"], s["category"], self.mode),
                                     self.norm_stats)
            cs = 512
            Hh, Ww = image.shape[1], image.shape[2]
            y0, x0 = (Hh - cs) // 2, (Ww - cs) // 2
            image = image[:, y0:y0+cs, x0:x0+cs]
            mask = mask[y0:y0+cs, x0:x0+cs]
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def _augment_5band(self, image, mask):
        """5 波段反射率增强: 几何与 _augment_rgb 一致, gamma 作用于全部 5 波段."""
        aug = self.augmentor
        image, mask = aug._random_crop(image, mask)
        if np.random.rand() < aug.prob_hflip:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if np.random.rand() < aug.prob_vflip:
            image = image[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        if np.random.rand() < aug.prob_small_rotate:
            angle = np.random.uniform(-aug.rotate_range, aug.rotate_range)
            _, Hh, Ww = image.shape
            M = cv2.getRotationMatrix2D((Ww / 2, Hh / 2), angle, 1.0)
            image = np.stack([
                cv2.warpAffine(image[c], M, (Ww, Hh),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                for c in range(image.shape[0])], axis=0)
            mask = cv2.warpAffine(mask, M, (Ww, Hh),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        if np.random.rand() < aug.prob_scale:
            _, Hh, Ww = image.shape
            scale = np.random.uniform(*aug.scale_range)
            nh, nw = int(Hh * scale), int(Ww * scale)
            image = np.stack([cv2.resize(image[c], (nw, nh),
                                         interpolation=cv2.INTER_LINEAR)
                              for c in range(image.shape[0])], axis=0)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            y0, x0 = max(0, (nh - Hh) // 2), max(0, (nw - Ww) // 2)
            image = image[:, y0:y0+Hh, x0:x0+Ww]
            mask = mask[y0:y0+Hh, x0:x0+Ww]
            _, Hh2, Ww2 = image.shape
            ph, pw = Hh - Hh2, Ww - Ww2
            if ph > 0 or pw > 0:
                image = np.pad(image, ((0, 0), (0, max(0, ph)), (0, max(0, pw))),
                               mode="reflect")
                mask = np.pad(mask, ((0, max(0, ph)), (0, max(0, pw))), mode="reflect")
        if np.random.rand() < aug.prob_gamma:
            n = image.shape[0]
            gamma = np.random.uniform(*aug.gamma_range, size=(n, 1, 1)).astype(np.float32)
            image = np.clip(image, 0, 1) ** gamma
        if np.random.rand() < aug.prob_noise:
            image = image + np.random.normal(0, aug.noise_std, image.shape).astype(np.float32)
        if np.random.rand() < aug.prob_jitter:
            n = image.shape[0]
            image = image + np.random.normal(0, aug.jitter_std, size=(n, 1, 1)).astype(np.float32)
        return image.astype(np.float32), mask.astype(np.uint8)


@torch.no_grad()
def validate(model, dataset, criterion, device, norm_stats, mode, desc="[val]"):
    """滑窗验证整图 (与 train_rgb.py 的 validate 同构, 仅加载函数不同)."""
    model.eval()
    tracker = MetricTracker(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    total_loss, n = 0, 0
    nan_val = []
    cs = 512
    for s in tqdm(dataset.samples, desc=desc, dynamic_ncols=True, leave=False):
        image = normalize_tensor(load_indices_input(s["stem"], s["category"], mode),
                                 norm_stats)
        image_t = torch.from_numpy(image).float()
        mask_path = MASKS_DIR / s["category"] / f"{s['stem']}_mask.png"
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros((H, W), dtype=np.uint8)

        prob_map = sliding_window_predict(model, image_t, device,
                                          crop=cs, stride=cs // 2)
        pred = prob_map.argmax(dim=0).numpy()

        Hh, Ww = image.shape[1], image.shape[2]
        y0, x0 = (Hh - cs) // 2, (Ww - cs) // 2
        ic = image[:, y0:y0+cs, x0:x0+cs]
        gc = gt[y0:y0+cs, x0:x0+cs]
        inp = torch.from_numpy(ic).float().unsqueeze(0).to(device)
        tgt = torch.from_numpy(gc).long().unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(inp); loss, _ = criterion(logits, tgt)
        # v5.4a: 单场景 loss 非有限时跳过累加, 防止污染整轮 val_loss 均值
        if torch.isfinite(loss):
            total_loss += loss.item(); n += 1
        else:
            nan_val.append(s["stem"])

        tracker.update(pred, gt)
    m = tracker.compute()
    m["val_loss"] = total_loss / max(n, 1)
    if nan_val:
        print(f"  [警告] val 中 {len(nan_val)} 个场景 loss 非有限已跳过: {nan_val[:3]}")
    return m, tracker.confusion


@torch.no_grad()
def visualize_prediction(model, sample, norm_stats, mode, device, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import CLASS_COLORS_RGB
    model.eval()
    image = normalize_tensor(load_indices_input(sample["stem"], sample["category"], mode),
                            norm_stats)
    mask_path = MASKS_DIR / sample["category"] / f"{sample['stem']}_mask.png"
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image_t = torch.from_numpy(image).float()
    prob_map = sliding_window_predict(model, image_t, device, crop=512, stride=256)
    pred = prob_map.argmax(dim=0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rgb_vis = np.transpose(image[0:3], (1, 2, 0))  # [B,G,R] -> 显示用
    rgb_vis = rgb_vis[..., ::-1]
    rgb_vis = np.clip((rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-6), 0, 1)
    axes[0].imshow(rgb_vis); axes[0].set_title("RGB (base)")
    axes[0].axis("off")
    for idx, (msk, t) in enumerate([(gt, "GT"), (pred, "Pred")]):
        mc = np.zeros((*msk.shape, 3), dtype=np.uint8)
        for ci, col in CLASS_COLORS_RGB.items():
            mc[msk == ci] = col
        axes[idx+1].imshow(mc); axes[idx+1].set_title(t); axes[idx+1].axis("off")
    fig.suptitle(f"[{mode}+indices] {sample['category']}/{sample['stem']}")
    plt.tight_layout(); plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()


def main():
    parser = argparse.ArgumentParser(description="光谱指数增强模型训练 (E2/E3)")
    parser.add_argument("--mode", choices=["ms", "rgb"], default="ms",
                        help="ms=E2 (10ch) | rgb=E3 (7ch)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.998)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2; args.batch_size = 2

    mode_cfg = MODES[args.mode]
    train_dir = OUTPUT_DIR / mode_cfg["dir"]
    train_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = train_dir / "latest_checkpoint.pth"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.mode}+indices] Device: {device} | Epochs: {args.epochs} | "
          f"BS: {args.batch_size} | LR: {args.lr} | Seed: {args.seed}")
    print(f"[{args.mode}+indices] Input: {mode_cfg['n_in']}ch + Sobel -> "
          f"{mode_cfg['model']} | indices: {INDEX_NAMES}")

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    norm_stats = load_norm_stats_indices(args.mode)
    assert len(norm_stats["mean"]) == mode_cfg["n_in"], \
        f"norm stats len {len(norm_stats['mean'])} != {mode_cfg['n_in']}"
    print(f"  norm_stats: {norm_stats.get('n_scenes', '?')} scenes "
          f"({mode_cfg['stats']}, train only)")

    train_data, val_data = split["train"], split["val"]
    test_data = split.get("test", [])
    if args.quick:
        train_data, val_data = train_data[:20], val_data[:8]
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    with open(train_dir / "config_used.json", "w") as f:
        json.dump({**vars(args), "indices": INDEX_NAMES,
                  "model_class": mode_cfg["model"]}, f, indent=2)

    augmentor = SegmentationAugmentor(crop_size=args.crop_size)
    train_ds = IndexDataset(train_data, norm_stats, augmentor=augmentor, mode=args.mode)
    val_ds = IndexDataset(val_data, norm_stats, augmentor=None, mode=args.mode)
    sampler = torch.utils.data.WeightedRandomSampler(
        compute_sample_weights(train_data), len(train_data), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=len(train_ds) >= args.batch_size)

    model_cls = {"ms": DeepLabV3PlusIndices,
                 "rgb": DeepLabV3PlusRGBIndices}[args.mode]
    model = model_cls(num_classes=NUM_CLASSES, backbone_pretrained=True,
                      output_stride=16).to(device)
    print(f"Model: {mode_cfg['model']} | params={model.count_parameters():,}")

    class_counts = [CLASS_PIXEL_COUNTS[c] for c in CLASS_NAMES]
    alpha = compute_focal_alpha(class_counts).to(device)
    criterion = CombinedLoss(alpha=alpha, ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.epochs,
                                      warmup_epochs=args.warmup_epochs,
                                      min_lr=args.min_lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_miou": [],
               "val_score": [], "lr": []}
    best_score, best_miou = -1.0, 0.0
    patience_counter = 0
    if args.resume and latest_ckpt.exists():
        print(f"\n[断点续训] 恢复自: {latest_ckpt}")
        ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        best_miou = ckpt.get("best_miou", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", history)
        print(f"[断点续训] 从 epoch {start_epoch + 1} 继续 | "
              f"best_score={best_score:.4f} | 已完成 {start_epoch} epochs")
        if start_epoch >= args.epochs:
            print(f"[断点续训] 已完成全部 {args.epochs} epochs. 如需重训: --no-resume")
            return

    print(f"\n{'='*70}")
    print(f"[{args.mode}+indices] 训练开始 | best model 依据: balance score")
    print(f"{'='*70}")

    metrics_meta = {"input_mode": f"{args.mode}_indices",
                    "model_class": mode_cfg["model"],
                    "indices": INDEX_NAMES}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, nan_skipped = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device, scaler, ema,
                                          grad_clip=args.grad_clip, epoch_idx=epoch + 1)
        vm, cm = validate(model, val_ds, criterion, device, norm_stats,
                         args.mode, desc=f"Ep {epoch+1:03d} [val]")
        scheduler.step()
        score = compute_balance_score(vm)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tl); history["val_loss"].append(vm["val_loss"])
        history["val_miou"].append(vm["mIoU"])
        history["val_score"].append(score); history["lr"].append(current_lr)

        if score > best_score:
            best_score = score
            best_miou = vm["mIoU"]
            tag = " >>> BEST"
            ema.apply_shadow()
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "miou": best_miou, "score": best_score,
                        "metrics": vm, **metrics_meta},
                       train_dir / "best_model.pth")
            ema.restore()
            patience_counter = 0
            vd = train_dir / "predictions"; vd.mkdir(exist_ok=True)
            for s in val_data[:3]:
                visualize_prediction(model, s, norm_stats, args.mode, device,
                                     vd / f"{s['stem']}_pred.png")
            plot_confusion(cm, train_dir / "confusion_matrix.png",
                           title=f"Confusion Matrix ({args.mode}+indices)")
        else:
            patience_counter += 1
            tag = f" (wait {patience_counter}/{args.patience})"

        with open(train_dir / "metrics.json", "w") as f:
            json.dump({"history": history, "best_miou": best_miou,
                       "best_score": best_score, "final_val": vm,
                       **metrics_meta, "test": None}, f, indent=2)
        plot_curves(history, train_dir / "training_curves.png")

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "ema_shadow": ema.shadow,
                    "best_score": best_score, "best_miou": best_miou,
                    "patience_counter": patience_counter,
                    "history": history}, latest_ckpt)

        print(f"Ep {epoch+1:03d}/{args.epochs} ({time.time()-t0:.0f}s) | "
              f"tl={tl:.4f} vl={vm['val_loss']:.4f} | mIoU={vm['mIoU']:.4f} "
              f"OA={vm['OA']:.4f} | score={score:.4f}{tag}")
        for c in DEFECT_CLASSES:
            print(f"    {c:25s} IoU={vm['per_class_IoU'][c]:.3f} "
                  f"F1={vm['per_class_F1'][c]:.3f} "
                  f"P={vm['per_class_Precision'][c]:.3f} "
                  f"R={vm['per_class_Recall'][c]:.3f}")

        if patience_counter >= args.patience and not args.quick:
            print(f"\n[早停] 连续 {args.patience} 轮 score 未提升, 停止训练.")
            break

    final_metrics = {}
    if test_data and not args.quick:
        test_ds = IndexDataset(test_data, norm_stats, augmentor=None, mode=args.mode)
        print(f"\n=== Test Set Evaluation ({args.mode}+indices) ===")
        test_m, test_cm = validate(model, test_ds, criterion, device, norm_stats,
                                   args.mode, desc="[test]")
        test_score = compute_balance_score(test_m)
        print(f"Test: mIoU={test_m['mIoU']:.4f} OA={test_m['OA']:.4f} "
              f"score={test_score:.4f}")
        for c in DEFECT_CLASSES:
            print(f"    {c:25s} IoU={test_m['per_class_IoU'][c]:.3f} "
                  f"F1={test_m['per_class_F1'][c]:.3f}")
        final_metrics["test"] = test_m
        plot_confusion(test_cm, train_dir / "test_confusion_matrix.png",
                       title=f"Test Confusion Matrix ({args.mode}+indices)")
        with open(train_dir / "metrics.json", "w") as f:
            json.dump({"history": history, "best_miou": best_miou,
                       "best_score": best_score, "final_val": vm,
                       **metrics_meta, "test": test_m}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[{args.mode}+indices] 训练完成 | Best score: {best_score:.4f} "
          f"| Best mIoU: {best_miou:.4f}")
    print(f"Best model: {train_dir / 'best_model.pth'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
