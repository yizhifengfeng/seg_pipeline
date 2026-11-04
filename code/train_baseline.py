# -*- coding: utf-8 -*-
"""
基线模型训练(仅 pan 输入) — v2 工程增强版 + v4 损失
===================================================
python train_baseline.py --quick              # 快速验证(2 epoch)
python train_baseline.py --epochs 80 --batch_size 4 --lr 0.001  # v4 默认 80 ep
python train_baseline.py --no-resume          # 忽略断点从头训练

与 train.py 的唯一差异:
  - 数据加载: load_pan_only (1 通道) vs load_6channel (6 通道)
  - 归一化: norm_stats_pan.json (pan only) vs norm_stats.json (6 bands)
  - 模型: DeepLabV3PlusBaseline (2ch 输入) vs DeepLabV3Plus (8ch 输入)

v4 调整 (2026-11-04, 与 train.py 完全同步以保证对比公平):
  - epochs 默认 50 -> 80
  - patience 默认 15 -> 20
  - Focal alpha 默认走 losses.py 的 v4 权重 [0.5, 0.90, 0.65, 1.17, 2.30]

v2 增强 (与 train.py 完全一致, 保证对比公平):
  tqdm 进度条 / 断点续训 / balance score 评分 / WarmupCosine / EMA /
  梯度裁剪 / NaN 保险丝 / 实时 metrics.json / 配置 dump

输出到 outputs/train_baseline/.
"""
import json
import sys
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

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
LATEST_CKPT = TRAIN_DIR / "latest_checkpoint.pth"
DEFECT_CLASSES = ["crack", "biological_attachment", "flaking", "hollow"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Warmup + Cosine LR 调度 (v2, 与 train.py 一致) ──────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = max(1, warmup_epochs)
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.last_epoch = 0

    def _compute_lr(self, base_lr):
        if self.last_epoch <= self.warmup_epochs:
            return base_lr * self.last_epoch / self.warmup_epochs
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr - self.min_lr) * cosine

    def step(self):
        self.last_epoch += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self._compute_lr(self.base_lrs[i])

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = state.get("last_epoch", 0)
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self._compute_lr(self.base_lrs[i])


# ── EMA (v2, 与 train.py 一致) ─────────────────────────────
class EMA:
    def __init__(self, model: nn.Module, decay=0.998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name]
                                     + (1.0 - self.decay) * param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def to_device(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}


class PanDataset(Dataset):
    """基线数据集: 仅加载 pan 单通道.

    增强流程: load_pan_only(原始) -> augment(几何+noise/jitter) -> normalize.
    pan 是 DN 不是反射率, 不做 gamma; noise/jitter 对 pan 安全.
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
            tensor, mask = self._augment_pan(tensor, mask)
            image = normalize_pan_tensor(tensor, self.norm_stats)
        else:
            image = normalize_pan_tensor(tensor, self.norm_stats)
            cs = 512
            Hh, Ww = image.shape[1], image.shape[2]
            y0, x0 = (Hh - cs) // 2, (Ww - cs) // 2
            image = image[:, y0:y0+cs, x0:x0+cs]
            mask = mask[y0:y0+cs, x0:x0+cs]
        # v2: NaN/Inf 兜底
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
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
    """与 train.py 完全一致 (向量化 + ignore_index + v2 P/R)."""

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
            "per_class_Precision": {CLASS_NAMES[i]: float(prec[i]) for i in range(self.nc)},
            "per_class_Recall": {CLASS_NAMES[i]: float(rec[i]) for i in range(self.nc)},
        }


def compute_balance_score(metrics):
    """与 train.py 完全一致 (对比公平)."""
    m = metrics
    miou = float(m["mIoU"])
    f1 = np.mean([m["per_class_F1"][c] for c in DEFECT_CLASSES])
    p = np.mean([m["per_class_Precision"][c] for c in DEFECT_CLASSES])
    r = np.mean([m["per_class_Recall"][c] for c in DEFECT_CLASSES])
    balance_penalty = np.mean([abs(m["per_class_Precision"][c] - m["per_class_Recall"][c])
                               for c in DEFECT_CLASSES])
    return float(0.40 * miou + 0.30 * f1 + 0.15 * p + 0.15 * r - 0.10 * balance_penalty)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, ema=None, grad_clip=1.0, epoch_idx=0):
    model.train()
    total_loss, n = 0.0, 0
    nan_skipped = 0
    pbar = tqdm(loader, desc=f"Ep {epoch_idx:03d} [train]",
                dynamic_ncols=True, leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(images); loss, parts = criterion(logits, masks)
        else:
            logits = model(images); loss, parts = criterion(logits, masks)

        if not torch.isfinite(loss):
            nan_skipped += 1
            pbar.set_postfix(loss="NaN(SKIP)")
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema:
            ema.update()

        total_loss += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    if nan_skipped:
        print(f"  [警告] 本 epoch 跳过 {nan_skipped} 个 NaN batch")
    return total_loss / max(n, 1), nan_skipped


@torch.no_grad()
def validate(model, dataset, criterion, device, norm_stats, desc="[val]"):
    """滑窗验证整图 (与 train.py 一致, v2: tqdm)."""
    model.eval()
    tracker = MetricTracker(5, ignore_index=IGNORE_INDEX)
    total_loss, n = 0, 0
    cs = 512
    for s in tqdm(dataset.samples, desc=desc, dynamic_ncols=True, leave=False):
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
    print("\n=== Test Set Evaluation (Baseline) ===")
    m, cm = validate(model, dataset, criterion, device, norm_stats, desc="[test]")
    score = compute_balance_score(m)
    print(f"Test: mIoU={m['mIoU']:.4f} OA={m['OA']:.4f} score={score:.4f}")
    for c in DEFECT_CLASSES:
        print(f"    {c:25s} IoU={m['per_class_IoU'][c]:.3f} F1={m['per_class_F1'][c]:.3f} "
              f"P={m['per_class_Precision'][c]:.3f} R={m['per_class_Recall'][c]:.3f}")
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
    pan_vis = np.clip(image[0], -3, 3) / 3 * 0.5 + 0.5
    axes[0].imshow(pan_vis, cmap="gray"); axes[0].set_title("Pan (input)")
    axes[0].axis("off")
    for idx, (msk, t) in enumerate([(gt, "GT"), (pred, "Pred")]):
        # v2 修复: uint8 避免 "Clipping input data" 警告
        mc = np.zeros((*msk.shape, 3), dtype=np.uint8)
        for ci, col in CLASS_COLORS_RGB.items():
            mc[msk == ci] = col
        axes[idx+1].imshow(mc); axes[idx+1].set_title(t); axes[idx+1].axis("off")
    fig.suptitle(f"[Baseline] {sample['category']}/{sample['stem']}")
    plt.tight_layout(); plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()


def plot_confusion(cm, path, title="Confusion Matrix (Baseline)"):
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
    ax.set_title(title); fig.colorbar(im)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_curves(history, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    a1 = axes[0]
    a1.plot(history["train_loss"], label="train"); a1.plot(history["val_loss"], label="val")
    a1.set_title("Loss (Baseline)"); a1.set_xlabel("Epoch"); a1.legend(); a1.grid(alpha=0.3)
    a2 = axes[1]
    a2.plot(history["val_miou"], label="mIoU", color="green")
    a2.set_title("Val mIoU (Baseline)"); a2.set_xlabel("Epoch"); a2.legend(); a2.grid(alpha=0.3)
    a3 = axes[2]
    if history.get("val_score"):
        a3.plot(history["val_score"], label="balance score", color="purple")
        a3.set_title("Val Balance Score (Baseline)"); a3.set_xlabel("Epoch")
        a3.legend(); a3.grid(alpha=0.3)
    if history.get("lr"):
        a3.plot(history["lr"], label="LR", color="brown", linestyle="--")
        a3.set_yscale("log")
        a3.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def main():
    parser = argparse.ArgumentParser(description="基线模型训练 (v2: 进度条+断点续训+balance score)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="完整训练 epoch 数 (v4 默认 80; v2 旧默认 50)")
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
    parser.add_argument("--patience", type=int, default=20,
                        help="早停 patience (v4 默认 20; v2 旧默认 15)")
    args = parser.parse_args()
    if args.quick: args.epochs = 2; args.batch_size = 2

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BASELINE] Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | "
          f"LR: {args.lr} | Seed: {args.seed}")
    print(f"[BASELINE] Input: pan only (1 channel + 1 edge = 2ch)")
    print(f"v2: resume={args.resume} | warmup={args.warmup_epochs} | "
          f"grad_clip={args.grad_clip} | EMA decay={args.ema_decay}")
    print(f"IGNORE_INDEX: {IGNORE_INDEX}")

    with open(SPLIT_PATH) as f: split = json.load(f)
    # 尝试加载 pan-only norm_stats, 若不存在则回退
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

    # v2: 保存本次训练配置
    with open(TRAIN_DIR / "config_used.json", "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)

    augmentor = SegmentationAugmentor(crop_size=args.crop_size, post_normalized=True)
    train_ds = PanDataset(train_data, norm_stats, augmentor=augmentor)
    val_ds = PanDataset(val_data, norm_stats, augmentor=None)
    sampler = torch.utils.data.WeightedRandomSampler(
        compute_sample_weights(train_data), len(train_data), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=len(train_ds) >= args.batch_size)

    model = DeepLabV3PlusBaseline(num_classes=5, backbone_pretrained=True,
                                  output_stride=16).to(device)
    print(f"Model: DeepLabV3PlusBaseline | params={model.count_parameters():,}")

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

    # ── v2: 断点续训 ──────────────────────────────────────
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_miou": [],
               "val_score": [], "lr": []}
    best_score, best_miou = -1.0, 0.0
    patience_counter = 0
    if args.resume and LATEST_CKPT.exists():
        print(f"\n[断点续训] 恢复自: {LATEST_CKPT}")
        ckpt = torch.load(str(LATEST_CKPT), map_location=device, weights_only=False)
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
    print(f"[BASELINE] 训练开始 | best model 依据: balance score")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, nan_skipped = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device, scaler, ema,
                                          grad_clip=args.grad_clip, epoch_idx=epoch + 1)
        vm, cm = validate(model, val_ds, criterion, device, norm_stats,
                          desc=f"Ep {epoch+1:03d} [val]")
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
                        "metrics": vm,
                        "input_mode": "pan_only"},
                       TRAIN_DIR / "best_model.pth")
            ema.restore()
            patience_counter = 0
            vd = TRAIN_DIR / "predictions"; vd.mkdir(exist_ok=True)
            for s in val_data[:3]:
                visualize_prediction(model, s, norm_stats, device,
                                     vd / f"{s['stem']}_pred.png")
            plot_confusion(cm, TRAIN_DIR / "confusion_matrix.png")
        else:
            patience_counter += 1
            tag = f" (wait {patience_counter}/{args.patience})"

        with open(TRAIN_DIR / "metrics.json", "w") as f:
            json.dump({"history": history, "best_miou": best_miou,
                       "best_score": best_score, "final_val": vm,
                       "input_mode": "pan_only",
                       "model_class": "DeepLabV3PlusBaseline",
                       "test": None}, f, indent=2)
        plot_curves(history, TRAIN_DIR / "training_curves.png")

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "ema_shadow": ema.shadow,
                    "best_score": best_score, "best_miou": best_miou,
                    "patience_counter": patience_counter,
                    "history": history}, LATEST_CKPT)

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
        test_ds = PanDataset(test_data, norm_stats, augmentor=None)
        test_m, test_cm = evaluate_test(model, test_ds, criterion, device, norm_stats)
        final_metrics["test"] = test_m
        plot_confusion(test_cm, TRAIN_DIR / "test_confusion_matrix.png",
                       title="Test Confusion Matrix (Baseline)")
        with open(TRAIN_DIR / "metrics.json", "w") as f:
            json.dump({"history": history, "best_miou": best_miou,
                       "best_score": best_score, "final_val": vm,
                       "input_mode": "pan_only",
                       "model_class": "DeepLabV3PlusBaseline",
                       "test": test_m}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[BASELINE] 训练完成 | Best score: {best_score:.4f} | Best mIoU: {best_miou:.4f}")
    if final_metrics.get("test"):
        print(f"Test: mIoU={final_metrics['test']['mIoU']:.4f} "
              f"score={compute_balance_score(final_metrics['test']):.4f}")
    print(f"Best model: {TRAIN_DIR / 'best_model.pth'}")
    print(f"断点文件:   {LATEST_CKPT} (可续训)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
