# -*- coding: utf-8 -*-
"""
基线模型训练(仅 pan 输入) — 对比实验用
========================================
python train_baseline.py --quick              # 快速验证(2 epoch)
python train_baseline.py --epochs 50 --batch_size 4 --lr 0.001

与 train.py 的唯一差异:
  - 数据加载: load_pan_only (1 通道) vs load_6channel (6 通道)
  - 归一化: norm_stats_pan.json (pan only) vs norm_stats.json (6 bands)
  - 模型: DeepLabV3PlusBaseline (2ch 输入) vs DeepLabV3Plus (8ch 输入)
  - 增强几何部分完全一致; 光谱增强(gamma)对 pan 无效(pan 是 DN 不是反射率),
    noise/jitter 仍适用

其余(损失/优化器/调度/验证/滑窗/test评估/可视化)与 train.py 完全一致.
输出到 outputs/train_baseline/.
"""
import json
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (MASKS_DIR, SPLIT_PATH, OUTPUT_DIR, CONFIGS_DIR,
                    CLASS_NAMES, H, W, IGNORE_INDEX, CLASS_PIXEL_COUNTS,
                    CLASS_COLORS_RGB)
from build_input_tensor import load_pan_only, normalize_pan_tensor, load_norm_stats_pan
from augment import SegmentationAugmentor, compute_sample_weights
from model import DeepLabV3PlusBaseline
from losses import CombinedLoss, compute_focal_alpha
from predict_utils import sliding_window_predict

TRAIN_DIR = OUTPUT_DIR / "train_baseline"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PanDataset(Dataset):
    """基线数据集: 仅加载 pan 单通道.

    增强流程: load_pan_only(原始) -> augment(几何+noise/jitter) -> normalize.
    注意: gamma 增强对 pan(DN 0-65535)无意义, augmentor 的 post_normalized=False
    模式下 gamma 只作用于 channel 0-4(反射率), pan(channel 5)跳过 gamma.
    但这里只有 1 个通道(pan), gamma 不会触发(augmentor 检查 channel 数).
    为安全起见, 我们在 augment 后手动跳过 gamma: 直接用 post_normalized=True 模式
    (只做 noise/jitter, 对 pan 安全).
    """

    def __init__(self, split_data, norm_stats_pan, augmentor=None):
        self.samples = split_data
        self.norm_stats = norm_stats_pan
        self.augmentor = augmentor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tensor = load_pan_only(s["stem"], s["category"])  # (1, H, W) 原始 pan
        mask_path = MASKS_DIR / s["category"] / f"{s['stem']}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)
        if self.augmentor:
            # pan 是单通道, augmentor 的 _small_rotate/_random_crop/_random_scale
            # 对 (C,H,W) 通用, C=1 也工作. gamma 在 post_normalized=True 时跳过.
            # 但 augmentor 内部有 image[:5] 切片(gamma 对 b1-b5), 对 (1,H,W) 会 IndexError.
            # 解决: 把 pan 复制到 6 通道做增强, 再取回第 0 通道(pan 位置).
            # 更简洁: 直接手动做几何增强 + noise.
            tensor, mask = self._augment_pan(tensor, mask)
            image = normalize_pan_tensor(tensor, self.norm_stats)
        else:
            image = normalize_pan_tensor(tensor, self.norm_stats)
            cs = 512
            Hh, Ww = image.shape[1], image.shape[2]
            y0, x0 = (Hh - cs) // 2, (Ww - cs) // 2
            image = image[:, y0:y0+cs, x0:x0+cs]
            mask = mask[y0:y0+cs, x0:x0+cs]
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def _augment_pan(self, image, mask):
        """pan 专用增强: 几何(翻转/旋转/缩放/裁剪) + noise/jitter.
        不做 gamma(pan 是 DN 不是反射率).
        """
        aug = self.augmentor
        # 1. 随机裁剪
        image, mask = aug._random_crop(image, mask)
        # 2. 水平翻转
        if np.random.rand() < aug.prob_hflip:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        # 3. 垂直翻转(低概率)
        if np.random.rand() < aug.prob_vflip:
            image = image[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        # 4. 小角度旋转
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
        # 5. 随机缩放 (pan 专用, augmentor 的 _random_scale 硬编码了 6 通道)
        if np.random.rand() < aug.prob_scale:
            _, Hh, Ww = image.shape
            scale = np.random.uniform(*aug.scale_range)
            nh, nw = int(Hh * scale), int(Ww * scale)
            image = np.stack([cv2.resize(image[c], (nw, nh),
                                         interpolation=cv2.INTER_LINEAR)
                              for c in range(image.shape[0])], axis=0)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            # center crop or pad 回原尺寸
            y0, x0 = max(0, (nh - Hh) // 2), max(0, (nw - Ww) // 2)
            image = image[:, y0:y0+Hh, x0:x0+Ww]
            mask = mask[y0:y0+Hh, x0:x0+Ww]
            _, Hh2, Ww2 = image.shape
            ph, pw = Hh - Hh2, Ww - Ww2
            if ph > 0 or pw > 0:
                image = np.pad(image, ((0, 0), (0, max(0, ph)), (0, max(0, pw))),
                               mode="reflect")
                mask = np.pad(mask, ((0, max(0, ph)), (0, max(0, pw))), mode="reflect")
        # 6. noise + jitter (对 pan DN 安全)
        if np.random.rand() < aug.prob_noise:
            image = image + np.random.normal(0, aug.noise_std, image.shape).astype(np.float32)
        if np.random.rand() < aug.prob_jitter:
            image = image + np.random.normal(0, aug.jitter_std, size=(1, 1, 1)).astype(np.float32)
        return image.astype(np.float32), mask.astype(np.uint8)


class MetricTracker:
    """与 train.py 完全一致 (向量化, ignore_index 支持)."""

    def __init__(self, num_classes=5, ignore_index=IGNORE_INDEX):
        self.nc = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred = pred.astype(np.int64).ravel()
        target = target.astype(np.int64).ravel()
        valid = (target >= 0) & (target < self.nc) & (pred >= 0) & (pred < self.nc)
        if not valid.any():
            return
        t = target[valid]
        p = pred[valid]
        idx = t * self.nc + p
        bins = np.bincount(idx, minlength=self.nc * self.nc)
        self.confusion += bins.reshape(self.nc, self.nc)

    def compute(self):
        c = self.confusion
        tp = np.diag(c); fp = c.sum(0) - tp; fn = c.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-6)
        return {
            "mIoU": float(np.nanmean(iou)), "OA": float(tp.sum() / max(c.sum(), 1)),
            "per_class_IoU": {CLASS_NAMES[i]: float(iou[i]) for i in range(self.nc)},
            "per_class_F1": {CLASS_NAMES[i]: float(f1[i]) for i in range(self.nc)},
        }


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, n = 0, 0
    parts_acc = {"focal": 0, "lovasz": 0, "boundary": 0}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(images); loss, parts = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(images); loss, parts = criterion(logits, masks)
            loss.backward(); optimizer.step()
        total_loss += loss.item(); n += 1
        for k in parts_acc: parts_acc[k] += parts[k]
    return total_loss / max(n, 1), {k: v / max(n, 1) for k, v in parts_acc.items()}


@torch.no_grad()
def validate(model, dataset, criterion, device, norm_stats):
    """滑窗验证整图 (与 train.py 一致)."""
    model.eval()
    tracker = MetricTracker(5, ignore_index=IGNORE_INDEX)
    total_loss, n = 0, 0
    cs = 512
    for s in dataset.samples:
        tensor = load_pan_only(s["stem"], s["category"])
        image = normalize_pan_tensor(tensor, norm_stats)
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
        total_loss += loss.item(); n += 1

        tracker.update(pred, gt)
    m = tracker.compute(); m["val_loss"] = total_loss / max(n, 1)
    return m, tracker.confusion


@torch.no_grad()
def evaluate_test(model, dataset, criterion, device, norm_stats):
    """在 test set 上跑最终评估 (与 train.py 一致)."""
    print("\n=== Test Set Evaluation (Baseline) ===")
    m, cm = validate(model, dataset, criterion, device, norm_stats)
    print(f"Test: mIoU={m['mIoU']:.4f} OA={m['OA']:.4f}")
    for c in ["crack", "biological_attachment", "flaking", "hollow"]:
        print(f"    {c:25s} IoU={m['per_class_IoU'][c]:.3f} F1={m['per_class_F1'][c]:.3f}")
    return m, cm


@torch.no_grad()
def visualize_prediction(model, sample, norm_stats, device, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    tensor = load_pan_only(sample["stem"], sample["category"])
    image = normalize_pan_tensor(tensor, norm_stats)
    mask_path = MASKS_DIR / sample["category"] / f"{sample['stem']}_mask.png"
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image_t = torch.from_numpy(image).float()
    prob_map = sliding_window_predict(model, image_t, device, crop=512, stride=256)
    pred = prob_map.argmax(dim=0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # pan 可视化(灰度)
    pan_vis = np.clip(image[0], -3, 3) / 3 * 0.5 + 0.5
    axes[0].imshow(pan_vis, cmap="gray"); axes[0].set_title("Pan (input)")
    axes[0].axis("off")
    for idx, (msk, t) in enumerate([(gt, "GT"), (pred, "Pred")]):
        mc = np.zeros((*msk.shape, 3))
        for ci, col in CLASS_COLORS_RGB.items():
            mc[msk == ci] = col
        axes[idx+1].imshow(mc); axes[idx+1].set_title(t); axes[idx+1].axis("off")
    fig.suptitle(f"[Baseline] {sample['category']}/{sample['stem']}")
    plt.tight_layout(); plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()


def plot_confusion(cm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    cm = cm.astype(float); cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j] > 0.5 else "black", fontsize=8)
    ax.set_title("Confusion Matrix (Baseline)"); fig.colorbar(im)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_curves(history, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(history["train_loss"], label="train"); a1.plot(history["val_loss"], label="val")
    a1.set_title("Loss (Baseline)"); a1.legend()
    a2.plot(history["val_miou"], label="mIoU", color="green")
    a2.set_title("Val mIoU (Baseline)"); a2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def main():
    parser = argparse.ArgumentParser(description="基线模型训练 (pan only)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick: args.epochs = 2; args.batch_size = 2

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BASELINE] Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | "
          f"LR: {args.lr} | Seed: {args.seed}")
    print(f"[BASELINE] Input: pan only (1 channel + 1 edge = 2ch)")
    print(f"IGNORE_INDEX: {IGNORE_INDEX}")

    with open(SPLIT_PATH) as f: split = json.load(f)
    # 尝试加载 pan-only norm_stats, 若不存在则提示运行 recompute_norm_stats_pan.py
    pan_stats_path = CONFIGS_DIR / "norm_stats_pan.json"
    if not pan_stats_path.exists():
        print(f"\n!! {pan_stats_path} not found.")
        print(f"!! Run first: python recompute_norm_stats_pan.py")
        print(f"!! Falling back to extracting pan stats from norm_stats.json (index 5)")
        with open(CONFIGS_DIR / "norm_stats.json") as f:
            full_stats = json.load(f)
        norm_stats = {
            "mean": [full_stats["mean"][5]],
            "std": [full_stats["std"][5]],
            "band_names": ["pan"],
            "n_scenes": full_stats.get("n_scenes", "?"),
            "note": "extracted from norm_stats.json index 5 (may have leakage)",
        }
    else:
        with open(pan_stats_path) as f:
            norm_stats = json.load(f)
        print(f"  Pan norm_stats: {norm_stats.get('n_scenes', '?')} scenes (train only)")

    train_data, val_data = split["train"], split["val"]
    test_data = split.get("test", [])
    if args.quick: train_data, val_data = train_data[:20], val_data[:8]
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    augmentor = SegmentationAugmentor(crop_size=args.crop_size, post_normalized=True)
    train_ds = PanDataset(train_data, norm_stats, augmentor=augmentor)
    val_ds = PanDataset(val_data, norm_stats, augmentor=None)
    sampler = torch.utils.data.WeightedRandomSampler(
        compute_sample_weights(train_data), len(train_data), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)

    model = DeepLabV3PlusBaseline(num_classes=5, backbone_pretrained=True,
                                  output_stride=16).to(device)
    print(f"Model: DeepLabV3PlusBaseline | params={model.count_parameters():,}")

    class_counts = [CLASS_PIXEL_COUNTS[c] for c in CLASS_NAMES]
    alpha = compute_focal_alpha(class_counts).to(device)
    criterion = CombinedLoss(alpha=alpha, ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou, bad, patience = 0, 0, 10
    for epoch in range(args.epochs):
        t0 = time.time()
        tl, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vm, cm = validate(model, val_ds, criterion, device, norm_stats)
        scheduler.step()
        history["train_loss"].append(tl); history["val_loss"].append(vm["val_loss"])
        history["val_miou"].append(vm["mIoU"])
        print(f"Ep {epoch+1}/{args.epochs} ({time.time()-t0:.0f}s) | "
              f"tl={tl:.4f} vl={vm['val_loss']:.4f} mIoU={vm['mIoU']:.4f} OA={vm['OA']:.4f}")
        for c in ["crack","biological_attachment","flaking","hollow"]:
            print(f"    {c:25s} IoU={vm['per_class_IoU'][c]:.3f} F1={vm['per_class_F1'][c]:.3f}")
        if vm["mIoU"] > best_miou:
            best_miou = vm["mIoU"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "miou": best_miou,
                        "input_mode": "pan_only"},
                       TRAIN_DIR / "best_model.pth")
            bad = 0
            vd = TRAIN_DIR / "predictions"; vd.mkdir(exist_ok=True)
            for s in val_data[:3]:
                visualize_prediction(model, s, norm_stats, device, vd / f"{s['stem']}_pred.png")
            plot_confusion(cm, TRAIN_DIR / "confusion_matrix.png")
        else:
            bad += 1
            if bad >= patience and not args.quick:
                print(f"Early stopping at epoch {epoch+1}"); break

    # Test set 评估
    final_metrics = {}
    if test_data and not args.quick:
        test_ds = PanDataset(test_data, norm_stats, augmentor=None)
        test_m, test_cm = evaluate_test(model, test_ds, criterion, device, norm_stats)
        final_metrics["test"] = test_m
        plot_confusion(test_cm, TRAIN_DIR / "test_confusion_matrix.png")

    with open(TRAIN_DIR / "metrics.json", "w") as f:
        json.dump({"history": history, "best_miou": best_miou,
                   "final_val": vm, "test": final_metrics.get("test", None),
                   "input_mode": "pan_only",
                   "model_class": "DeepLabV3PlusBaseline"}, f, indent=2)
    plot_curves(history, TRAIN_DIR / "training_curves.png")
    print(f"\n[BASELINE] Best Val mIoU: {best_miou:.4f}")
    if final_metrics.get("test"):
        print(f"[BASELINE] Test mIoU: {final_metrics['test']['mIoU']:.4f}")
    print(f"[BASELINE] Model: {TRAIN_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
