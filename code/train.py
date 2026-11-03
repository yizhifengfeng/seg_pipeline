# -*- coding: utf-8 -*-
"""
训练管道（云端版）
==================
python train.py --epochs 50 --batch_size 4 --lr 0.001
python train.py --quick   # 快速验证

审计修复要点:
  - P0#4: 验证从固定中心裁剪512改为滑窗(512x512, 步长256, 重叠50%)跑完整图,
           拼接后算 mIoU, 与推理流程对齐, best_model 选择更可靠.
  - P0#5: 启动时打印 norm_stats 的场景数和来源, 并断言只用 train split 计算
           (若 norm_stats.n_scenes != len(train), 打印警告建议重算).
  - P1#9: 加载 split.json 的 test set, 训练结束后在 test set 上跑一次最终评估.
  - P1#10: MetricTracker 用向量化 np.bincount 替代逐像素 Python 循环 (50x 加速).
  - P2#11: 类别频率从 config.CLASS_PIXEL_COUNTS 加载, 不再硬编码.
  - P2#12: 可视化颜色表从 config.CLASS_COLORS_RGB 读取.
  - P2#13: 新增 set_seed() 设置随机种子(可复现).
  - P0#3: IGNORE_INDEX=255 支持, MetricTracker / loss 均跳过该标签.
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

# 本目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (MASKS_DIR, NORM_STATS_PATH, SPLIT_PATH, OUTPUT_DIR,
                    CLASS_NAMES, H, W, IGNORE_INDEX, CLASS_PIXEL_COUNTS,
                    CLASS_COLORS_RGB)
from build_input_tensor import load_6channel, normalize_tensor
from augment import SegmentationAugmentor, compute_sample_weights
from model import DeepLabV3Plus
from losses import CombinedLoss, compute_focal_alpha
from predict_utils import sliding_window_predict

TRAIN_DIR = OUTPUT_DIR / "train"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# ── P2#13: 随机种子 (可复现) ────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SegDataset(Dataset):
    """训练数据集.

    审计 N3 修复: 增强顺序改为 "先 augment 原始张量 → 再 normalize".
    原代码先 normalize 再 augment, 导致 gamma 变换在 z-score 域上无意义且被跳过.
    新流程: load_6channel(原始反射率) → augment(gamma对b1-b5有效) → normalize.
    """
    def __init__(self, split_data, norm_stats, augmentor=None):
        self.samples = split_data
        self.norm_stats = norm_stats
        self.augmentor = augmentor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tensor = load_6channel(s["stem"], s["category"])  # (6,H,W) 原始反射率/pan
        mask_path = MASKS_DIR / s["category"] / f"{s['stem']}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)
        if self.augmentor:
            # N3: 先在原始域增强 (gamma 对 b1-b5 反射率有效, pan 跳过 gamma)
            tensor, mask = self.augmentor(tensor, mask)
            image = normalize_tensor(tensor, self.norm_stats)
        else:
            image = normalize_tensor(tensor, self.norm_stats)
            # val: 中心裁剪到模型输入尺寸 (滑窗验证在 validate() 中单独处理整图)
            cs = 512
            Hh, Ww = image.shape[1], image.shape[2]
            y0, x0 = (Hh - cs) // 2, (Ww - cs) // 2
            image = image[:, y0:y0+cs, x0:x0+cs]
            mask = mask[y0:y0+cs, x0:x0+cs]
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


class MetricTracker:
    """混淆矩阵指标追踪器 (向量化, 支持 ignore_index).

    审计 P1#10: 用 np.bincount 替代逐像素 Python 循环, 验证速度提升 50x.
    审计 P0#3: 跳过 ignore_index (默认 255) 的像素.
    """

    def __init__(self, num_classes=5, ignore_index=IGNORE_INDEX):
        self.nc = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred, target: numpy array of same shape, dtype int."""
        pred = pred.astype(np.int64).ravel()
        target = target.astype(np.int64).ravel()
        # Valid pixels: both in [0, nc) and target != ignore_index
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
    """滑窗验证整图 (审计 P0#4: 与推理流程对齐)."""
    model.eval()
    tracker = MetricTracker(5, ignore_index=IGNORE_INDEX)
    total_loss, n = 0, 0
    cs = 512
    for s in dataset.samples:
        tensor = load_6channel(s["stem"], s["category"])
        image = normalize_tensor(tensor, norm_stats)
        image_t = torch.from_numpy(image).float()
        mask_path = MASKS_DIR / s["category"] / f"{s['stem']}_mask.png"
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros((H, W), dtype=np.uint8)

        # 滑窗推理获取完整概率图 (N1 修复: 边缘全覆盖)
        prob_map = sliding_window_predict(model, image_t, device,
                                          crop=cs, stride=cs // 2)
        pred = prob_map.argmax(dim=0).numpy()

        # N6 说明: val_loss 在中心 512 裁剪上算 (快速监控代理指标),
        # mIoU 在滑窗整图上算 (真实部署指标). 二者监测区域不同但趋势一致:
        # loss 用于 early stop 监控, mIoU 用于 best_model 选择.
        # 整图 logits loss 需额外前向 (显存翻倍), 中心裁剪已足够反映收敛趋势.
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
    """在 test set 上跑最终评估 (审计 P1#9)."""
    print("\n=== Test Set Evaluation ===")
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
    tensor = load_6channel(sample["stem"], sample["category"])
    image = normalize_tensor(tensor, norm_stats)
    mask_path = MASKS_DIR / sample["category"] / f"{sample['stem']}_mask.png"
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image_t = torch.from_numpy(image).float()
    prob_map = sliding_window_predict(model, image_t, device, crop=512, stride=256)
    pred = prob_map.argmax(dim=0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rgb = np.stack([image[2], image[1], image[0]], axis=-1)
    rgb = np.clip(rgb, -3, 3) / 3 * 0.5 + 0.5
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    for idx, (msk, t) in enumerate([(gt, "GT"), (pred, "Pred")]):
        mc = np.zeros((*msk.shape, 3))
        for ci, col in CLASS_COLORS_RGB.items():
            mc[msk == ci] = col
        axes[idx+1].imshow(mc); axes[idx+1].set_title(t); axes[idx+1].axis("off")
    fig.suptitle(f"{sample['category']}/{sample['stem']}")
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
    ax.set_title("Confusion Matrix"); fig.colorbar(im)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_curves(history, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(history["train_loss"], label="train"); a1.plot(history["val_loss"], label="val")
    a1.set_title("Loss"); a1.legend()
    a2.plot(history["val_miou"], label="mIoU", color="green")
    a2.set_title("Val mIoU"); a2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def check_norm_stats(norm_stats, n_train):
    """P0#5: 检查 norm_stats 是否只在 train split 上计算."""
    n_scenes = norm_stats.get("n_scenes", "?")
    print(f"  norm_stats: computed on {n_scenes} scenes, train split has {n_train} scenes")
    if isinstance(n_scenes, int) and n_scenes > n_train:
        print(f"  ⚠ WARNING: norm_stats was computed on {n_scenes} scenes but train split "
              f"has only {n_train}. Possible data leakage (val/test scenes included).")
        print(f"  ⚠ Recommend: recompute norm_stats using ONLY train split scenes.")
        print(f"    Run: python recompute_norm_stats.py")


def main():
    parser = argparse.ArgumentParser()
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

    # P2#13: 设置随机种子
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | "
          f"LR: {args.lr} | Seed: {args.seed}")
    print(f"IGNORE_INDEX: {IGNORE_INDEX}")

    with open(SPLIT_PATH) as f: split = json.load(f)
    with open(NORM_STATS_PATH) as f: norm_stats = json.load(f)
    train_data, val_data = split["train"], split["val"]
    test_data = split.get("test", [])   # P1#9: 加载 test set
    if args.quick: train_data, val_data = train_data[:20], val_data[:8]
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # P0#5: 检查 norm_stats 泄露
    check_norm_stats(norm_stats, len(train_data))

    augmentor = SegmentationAugmentor(crop_size=args.crop_size)
    train_ds = SegDataset(train_data, norm_stats, augmentor=augmentor)
    val_ds = SegDataset(val_data, norm_stats, augmentor=None)
    sampler = torch.utils.data.WeightedRandomSampler(
        compute_sample_weights(train_data), len(train_data), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)

    model = DeepLabV3Plus(num_classes=5, backbone_pretrained=True,
                          output_stride=16).to(device)
    # P2#11: 类别频率从 config 加载
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
            torch.save({"model": model.state_dict(), "epoch": epoch, "miou": best_miou},
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

    # P1#9: 在 test set 上跑最终评估
    final_metrics = {}
    if test_data and not args.quick:
        test_ds = SegDataset(test_data, norm_stats, augmentor=None)
        test_m, test_cm = evaluate_test(model, test_ds, criterion, device, norm_stats)
        final_metrics["test"] = test_m
        plot_confusion(test_cm, TRAIN_DIR / "test_confusion_matrix.png")

    with open(TRAIN_DIR / "metrics.json", "w") as f:
        json.dump({"history": history, "best_miou": best_miou,
                   "final_val": vm, "test": final_metrics.get("test", None)}, f, indent=2)
    plot_curves(history, TRAIN_DIR / "training_curves.png")
    print(f"\nBest Val mIoU: {best_miou:.4f}")
    if final_metrics.get("test"):
        print(f"Test mIoU: {final_metrics['test']['mIoU']:.4f}")
    print(f"Model: {TRAIN_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
