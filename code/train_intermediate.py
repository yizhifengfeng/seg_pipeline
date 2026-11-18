# -*- coding: utf-8 -*-
"""
训练管道（多光谱中期融合 / 特征级融合版）— v6
====================================================
v6 变更 (基于 v5 结果诊断: 中期融合 110ep 未收敛, val_mIoU 到 ep107 仍单调爬升):
  - --epochs 默认 80 → 200 (双塔收敛慢 ~60%, 需要更长预算)
  - --min_lr 默认 1e-6 → 1e-5 (v5 中 lr 先于收敛耗尽, 退火终点过深)
  - --patience 默认 20 → 40 (配合更长训练, 避免早停误杀)
  - 新增 --tag: 输出目录后缀 (--tag narrow → outputs/train_intermediate_narrow/),
    用于窄版消融与主实验并行不互相覆盖

用法:
python train_intermediate.py --quick                                # 快速验证(2 epoch)
python train_intermediate.py --no-resume                            # v6 主实验 (200ep, 满宽)
python train_intermediate.py --no-resume --spec_width_scale 0.5 \
    --tag narrow                                                    # v6 窄版消融 (~32.6M, 回应容量混杂)

设计文档: lab/INTERMEDIATE_FUSION_PLAN.md

与 train.py (早期融合) 的差异 — 仅有 4 处, 其余逐行保留:
  1. 模型类: DeepLabV3Plus → DeepLabV3PlusIntermediateFusion
     (对称双塔 ResNet-34 + layer1-4 逐级 concat+1x1 融合; 输入接口仍是 (B,6,H,W),
      数据管道/损失/增强/评估流程与早期融合完全相同, 唯一变量=融合方式)
  2. 输出目录: outputs/train/ → outputs/train_intermediate/ (--tag 可加后缀)
     input_mode 标签: "multispectral_intermediate_fusion"
  3. 新增 --grad_accum: 梯度累积 (OOM 保险; 双塔显存约为单塔 1.8x,
     24GB 卡 batch=4 一般够用; 若 OOM 用 --batch_size 2 --grad_accum 2,
     等效 batch=4. 注意: 若启用, 早期融合也须用相同有效 batch 对照重训)
  4. 新增 --spec_width_scale: 光谱塔通道宽度 (1.0=主实验; 0.5=参数量消融,
     预训练权重沿通道维切片初始化; 消融版仅作参数量敏感性参考, 不作主结论)

其余全部与 train.py v4 一致: lr=1e-3, seed=42, v4 Focal alpha,
Warmup+Cosine, EMA, 梯度裁剪, NaN 保险丝, balance score 选优,
断点续训, 每 epoch 实时写 metrics.json.
注意: v6 重跑须加 --no-resume (旧 checkpoint 的 cosine 调度/ patience 状态基于 v5 配置).
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

# tqdm 有则用, 无则降级为普通迭代 (进度条 fallback)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# 本目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (MASKS_DIR, NORM_STATS_PATH, SPLIT_PATH, OUTPUT_DIR,
                    CLASS_NAMES, NUM_CLASSES, H, W, IGNORE_INDEX, CLASS_PIXEL_COUNTS,
                    CLASS_COLORS_RGB)
from build_input_tensor import load_6channel, normalize_tensor
from augment import SegmentationAugmentor, compute_sample_weights
from model import DeepLabV3PlusIntermediateFusion
from losses import CombinedLoss, compute_focal_alpha
from predict_utils import sliding_window_predict

TRAIN_DIR = OUTPUT_DIR / "train_intermediate"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
LATEST_CKPT = TRAIN_DIR / "latest_checkpoint.pth"
DEFECT_CLASSES = ["crack", "biological_attachment", "flaking"]
INPUT_MODE = "multispectral_intermediate_fusion"


# ── 随机种子 (可复现, 与 train.py 完全一致) ────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Warmup + Cosine LR 调度 (与 train.py 完全一致) ──────────
class WarmupCosineScheduler:
    """前 warmup_epochs 线性热身, 之后 cosine 退火到 min_lr."""

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


# ── EMA 权重指数移动平均 (与 train.py 完全一致) ─────────────
class EMA:
    """Exponential Moving Average of model weights.
    best_model 保存前 apply_shadow (用 EMA 平滑权重), 保存后 restore.
    """

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


# ── 数据集 (与 train.py 完全一致: 先 augment 原始张量再 normalize) ──
class SegDataset(Dataset):
    """训练数据集. 输入 (6,H,W) 与早期融合完全相同 — 模型内部再拆分模态."""

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
        # NaN/Inf 兜底 (脏数据防护)
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


# ── 指标追踪 (与 train.py 完全一致) ─────────────────────────
class MetricTracker:
    """混淆矩阵指标追踪器 (向量化, 支持 ignore_index, 含 per_class P/R)."""

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
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


# ── 综合评分标准 (与 train.py 完全一致) ─────────────────────
def compute_balance_score(metrics):
    """score = 0.40*mIoU + 0.30*meanDefectF1 + 0.15*meanDefectP + 0.15*meanDefectR
            - 0.10*meanDefect|P-R|  (仅缺陷类, 防 P-R 失衡)"""
    m = metrics
    miou = float(m["mIoU"])
    f1 = np.mean([m["per_class_F1"][c] for c in DEFECT_CLASSES])
    p = np.mean([m["per_class_Precision"][c] for c in DEFECT_CLASSES])
    r = np.mean([m["per_class_Recall"][c] for c in DEFECT_CLASSES])
    balance_penalty = np.mean([abs(m["per_class_Precision"][c] - m["per_class_Recall"][c])
                               for c in DEFECT_CLASSES])
    return float(0.40 * miou + 0.30 * f1 + 0.15 * p + 0.15 * r - 0.10 * balance_penalty)


# ── 训练单 epoch (v1 中期融合版: 新增梯度累积) ──────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, ema=None, grad_clip=1.0, epoch_idx=0,
                    grad_accum=1):
    """grad_accum>1 时: loss/grad_accum 累积, 每 grad_accum 个 batch step 一次,
    等效 batch_size = batch_size * grad_accum. epoch 末 flush 未满窗口.
    grad_accum=1 时与 train.py 的 train_one_epoch 行为完全一致."""
    model.train()
    total_loss, n = 0.0, 0
    nan_skipped = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Ep {epoch_idx:03d} [train]",
                dynamic_ncols=True, leave=False)

    def _optimizer_step():
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema:
            ema.update()

    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(images); loss, parts = criterion(logits, masks)
        else:
            logits = model(images); loss, parts = criterion(logits, masks)

        # NaN 保险丝: loss 非有限 → 跳过该 batch, 不更新权重
        if not torch.isfinite(loss):
            nan_skipped += 1
            pbar.set_postfix(loss="NaN(SKIP)")
            continue

        scaled_loss = loss / grad_accum
        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (i + 1) % grad_accum == 0:
            _optimizer_step()

        total_loss += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                         accum=grad_accum if grad_accum > 1 else None)

    # epoch 末 flush 未满的累积窗口 (drop_last=True 保证批数固定, 但通用起见仍处理)
    if grad_accum > 1 and n > 0 and n % grad_accum != 0:
        _optimizer_step()

    if nan_skipped:
        print(f"  [警告] 本 epoch 跳过 {nan_skipped} 个 NaN batch (数据可能含脏样本)")
    return total_loss / max(n, 1), nan_skipped


# ── 滑窗验证 (与 train.py 完全一致) ─────────────────────────
@torch.no_grad()
def validate(model, dataset, criterion, device, norm_stats, desc="[val]"):
    """滑窗验证整图 (与早期融合的评估流程完全一致, 保证指标可比)."""
    model.eval()
    tracker = MetricTracker(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    total_loss, n = 0.0, 0
    cs = 512
    for s in tqdm(dataset.samples, desc=desc, dynamic_ncols=True, leave=False):
        tensor = load_6channel(s["stem"], s["category"])
        image = normalize_tensor(tensor, norm_stats)
        image_t = torch.from_numpy(image).float()
        mask_path = MASKS_DIR / s["category"] / f"{s['stem']}_mask.png"
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros((H, W), dtype=np.uint8)

        prob_map = sliding_window_predict(model, image_t, device,
                                          crop=cs, stride=cs // 2)
        pred = prob_map.argmax(dim=0).numpy()

        # val_loss 在中心 512 裁剪上算 (快速监控代理指标),
        # mIoU 在滑窗整图上算 (真实部署指标).
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
    """在 test set 上跑最终评估 (与 train.py 完全一致)."""
    print("\n=== Test Set Evaluation ===")
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
        mc = np.zeros((*msk.shape, 3), dtype=np.uint8)
        for ci, col in CLASS_COLORS_RGB.items():
            mc[msk == ci] = col
        axes[idx+1].imshow(mc); axes[idx+1].set_title(t); axes[idx+1].axis("off")
    fig.suptitle(f"{sample['category']}/{sample['stem']}")
    plt.tight_layout(); plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()


def plot_confusion(cm, path, title="Confusion Matrix"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    cm = cm.astype(float); cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
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
    a1.set_title("Loss"); a1.set_xlabel("Epoch"); a1.legend(); a1.grid(alpha=0.3)
    a2 = axes[1]
    a2.plot(history["val_miou"], label="mIoU", color="green")
    a2.set_title("Val mIoU"); a2.set_xlabel("Epoch"); a2.legend(); a2.grid(alpha=0.3)
    a3 = axes[2]
    if history.get("val_score"):
        a3.plot(history["val_score"], label="balance score", color="purple")
        a3.set_title("Val Balance Score / LR"); a3.set_xlabel("Epoch")
        a3.legend(); a3.grid(alpha=0.3)
    if history.get("lr"):
        a3.plot(history["lr"], label="LR", color="brown", linestyle="--")
        a3.set_yscale("log")
        a3.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def check_norm_stats(norm_stats, n_train):
    """检查 norm_stats 是否只在 train split 上计算 (审计 P0#5, 两模型必须用同一份)."""
    n_scenes = norm_stats.get("n_scenes", "?")
    print(f"  norm_stats: computed on {n_scenes} scenes, train split has {n_train} scenes")
    if isinstance(n_scenes, int) and n_scenes > n_train:
        print(f"  ⚠ WARNING: norm_stats was computed on {n_scenes} scenes but train split "
              f"has only {n_train}. Possible data leakage (val/test scenes included).")
        print(f"  ⚠ Recommend: recompute norm_stats using ONLY train split scenes.")
        print(f"    Run: python recompute_norm_stats.py")


def main():
    parser = argparse.ArgumentParser(
        description="多光谱中期融合(特征级)模型训练 — 与 train.py v4 严格对齐, 唯一变量=融合方式")
    parser.add_argument("--epochs", type=int, default=200,
                        help="v6 默认 200 (双塔收敛慢 ~60%, v5 110ep 未收敛)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续训 (默认开启, 检测到 latest_checkpoint.pth 自动恢复)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="忽略已有断点, 从头训练")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="v6 默认 1e-5 (v5 中 1e-6 时 lr 先于收敛耗尽)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.998)
    parser.add_argument("--patience", type=int, default=40,
                        help="早停 patience (v6 默认 40, 配合 200ep 长训练)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="梯度累积步数 (OOM 保险: --batch_size 2 --grad_accum 2 "
                             "等效 batch 4; 启用时早期融合须用相同有效 batch 对照)")
    parser.add_argument("--spec_width_scale", type=float, default=1.0,
                        help="光谱塔通道宽度缩放 (1.0=主实验满宽; 0.5=参数量消融, "
                             "仅审稿需要时用)")
    parser.add_argument("--tag", default=None,
                        help="输出目录后缀: --tag narrow → outputs/train_intermediate_narrow/ "
                             "(默认无后缀, 写 outputs/train_intermediate/)。"
                             "消融实验必须加 tag, 避免覆盖主实验结果")
    args = parser.parse_args()
    if args.quick: args.epochs = 2; args.batch_size = 2

    # ── v6: tag 决定输出目录 (窄版消融与主实验隔离) ──
    global TRAIN_DIR, LATEST_CKPT
    if args.tag:
        TRAIN_DIR = OUTPUT_DIR / f"train_intermediate_{args.tag}"
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_CKPT = TRAIN_DIR / "latest_checkpoint.pth"

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | "
          f"LR: {args.lr} | Seed: {args.seed}")
    print(f"融合方式: INTERMEDIATE (dual-tower feature fusion) | "
          f"spec_width_scale={args.spec_width_scale} | grad_accum={args.grad_accum}")
    print(f"输出目录: {TRAIN_DIR}")
    print(f"v2 底座: resume={args.resume} | warmup={args.warmup_epochs} | "
          f"grad_clip={args.grad_clip} | EMA decay={args.ema_decay}")
    print(f"IGNORE_INDEX: {IGNORE_INDEX}")

    with open(SPLIT_PATH) as f: split = json.load(f)
    with open(NORM_STATS_PATH) as f: norm_stats = json.load(f)
    train_data, val_data = split["train"], split["val"]
    test_data = split.get("test", [])
    if args.quick: train_data, val_data = train_data[:20], val_data[:8]
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    check_norm_stats(norm_stats, len(split["train"]))  # 注意: 用完整 train split 数检查

    augmentor = SegmentationAugmentor(crop_size=args.crop_size)
    train_ds = SegDataset(train_data, norm_stats, augmentor=augmentor)
    val_ds = SegDataset(val_data, norm_stats, augmentor=None)
    sampler = torch.utils.data.WeightedRandomSampler(
        compute_sample_weights(train_data), len(train_data), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=len(train_ds) >= args.batch_size)

    model = DeepLabV3PlusIntermediateFusion(
        num_classes=NUM_CLASSES, backbone_pretrained=True, output_stride=16,
        spec_width_scale=args.spec_width_scale).to(device)
    n_params = model.count_parameters()
    print(f"Intermediate fusion model: {n_params:,} params ({n_params/1e6:.2f}M)")

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

    # 保存本次训练配置 + 模型元信息 (对比报告自动读取参数量)
    with open(TRAIN_DIR / "config_used.json", "w") as f:
        json.dump({**vars(args), "n_params": n_params,
                   "model_class": "DeepLabV3PlusIntermediateFusion",
                   "input_mode": INPUT_MODE,
                   "fusion": "intermediate"}, f, indent=2)

    # ── 断点续训 ──────────────────────────────────────────
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
            print(f"[断点续训] 已完成全部 {args.epochs} epochs, 无需继续. "
                  f"如需重新训练: --no-resume")
            return

    print(f"\n{'='*70}")
    print(f"训练开始 | 中期融合(特征级) | best model 依据: balance score = "
          f"0.40*mIoU + 0.30*defectF1 + 0.15*P + 0.15*R - 0.10*|P-R|")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, nan_skipped = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device, scaler, ema,
                                          grad_clip=args.grad_clip,
                                          epoch_idx=epoch + 1,
                                          grad_accum=args.grad_accum)
        vm, cm = validate(model, val_ds, criterion, device, norm_stats,
                          desc=f"Ep {epoch+1:03d} [val]")
        scheduler.step()
        score = compute_balance_score(vm)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tl); history["val_loss"].append(vm["val_loss"])
        history["val_miou"].append(vm["mIoU"])
        history["val_score"].append(score); history["lr"].append(current_lr)

        # ── best model 判断 (balance score) ────────────────
        if score > best_score:
            best_score = score
            best_miou = vm["mIoU"]
            tag = " >>> BEST"
            # EMA 权重保存 best (更平滑)
            ema.apply_shadow()
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "miou": best_miou, "score": best_score,
                        "metrics": vm,
                        "input_mode": INPUT_MODE,
                        "fusion": "intermediate",
                        "spec_width_scale": args.spec_width_scale,
                        "n_params": n_params},
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

        # ── 每 epoch 实时写 metrics.json (断电不丢) ─────────
        with open(TRAIN_DIR / "metrics.json", "w") as f:
            json.dump({"history": history, "best_miou": best_miou,
                       "best_score": best_score, "final_val": vm,
                       "input_mode": INPUT_MODE,
                       "fusion": "intermediate",
                       "spec_width_scale": args.spec_width_scale,
                       "n_params": n_params,
                       "test": None}, f, indent=2)
        plot_curves(history, TRAIN_DIR / "training_curves.png")

        # ── 每 epoch 保存断点 (断点续训) ─────────────────────
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

    # Test set 评估
    final_metrics = {}
    if test_data and not args.quick:
        test_ds = SegDataset(test_data, norm_stats, augmentor=None)
        test_m, test_cm = evaluate_test(model, test_ds, criterion, device, norm_stats)
        final_metrics["test"] = test_m
        plot_confusion(test_cm, TRAIN_DIR / "test_confusion_matrix.png",
                       title="Test Confusion Matrix")
        with open(TRAIN_DIR / "metrics.json", "w") as f:
            json.dump({"history": history, "best_miou": best_miou,
                       "best_score": best_score, "final_val": vm,
                       "input_mode": INPUT_MODE,
                       "fusion": "intermediate",
                       "spec_width_scale": args.spec_width_scale,
                       "n_params": n_params,
                       "test": test_m}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"训练完成 | Intermediate Fusion | Best balance score: {best_score:.4f} "
          f"| Best mIoU: {best_miou:.4f}")
    if final_metrics.get("test"):
        print(f"Test: mIoU={final_metrics['test']['mIoU']:.4f} "
              f"score={compute_balance_score(final_metrics['test']):.4f}")
    print(f"Best model: {TRAIN_DIR / 'best_model.pth'}")
    print(f"断点文件:   {LATEST_CKPT} (可续训)")
    print(f"指标日志:   {TRAIN_DIR / 'metrics.json'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
