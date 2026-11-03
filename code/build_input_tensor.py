# -*- coding: utf-8 -*-
"""
6 通道输入张量构建与归一化（云端版）
====================================
加载 b1-b5 + pan（重采样到 b2 网格），z-score 归一化。
PanAligner 已内联，不依赖外部文件。

Importable: from build_input_tensor import load_6channel, normalize_tensor, load_normalized
"""
import json
import re
import numpy as np
import tifffile
import cv2
from pathlib import Path

from config import (CALIBRATED_DIR, CONFIGS_DIR, MEDIAN_H_PATH,
                    NORM_STATS_PATH, H, W, BAND_NAMES)


# ---------------------------------------------------------------------------
# PanAligner（内联，原 pan_align_helper.py）
# ---------------------------------------------------------------------------
class PanAligner:
    """解析 pan_reg.txt，提供 b2 网格 -> pan 输出图坐标映射。"""

    def __init__(self, reg_txt_path):
        txt = Path(reg_txt_path).read_text(encoding="utf-8")
        hm = re.search(r"H=([-\d.eE+,]+)", txt)
        cm = re.search(r"crop=\(([-\d,]+)\)", txt)
        if not hm or not cm:
            raise ValueError(f"Cannot parse H/crop from {reg_txt_path}")
        self.H = np.array([float(v) for v in hm.group(1).split(",")]).reshape(3, 3)
        crop = [int(v) for v in cm.group(1).split(",")]
        self.x0, self.y0 = crop[0], crop[1]

    def b2_grid_to_pan_out(self, b2_w=W, b2_h=H):
        ys, xs = np.mgrid[0:b2_h, 0:b2_w]
        flat = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
        mapped = self.H @ flat
        mapped = mapped[:2] / mapped[2]
        PX = mapped[0].reshape(b2_h, b2_w) - self.x0
        PY = mapped[1].reshape(b2_h, b2_w) - self.y0
        return PX, PY


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------
def resample_pan_to_b2(pan_path, reg_txt_path):
    """把 pan 重采样到 b2 网格 (1456x1088)。fallback 场景用中位 H。

    审计修复:
      - P2#15: bare except (ValueError, Exception) 收窄为 (ValueError, KeyError, OSError),
               并打印 warning log, 不再静默吞掉 FileNotFoundError 等真实错误.
      - P2#18: cv2.remap 边界填充从 BORDER_CONSTANT(0) 改为 BORDER_REFLECT,
               避免硬黑边被模型当作伪特征.
    """
    import warnings
    pan = tifffile.imread(str(pan_path))  # uint16
    try:
        pa = PanAligner(reg_txt_path)
        PX, PY = pa.b2_grid_to_pan_out()
    except (ValueError, KeyError, OSError) as e:
        # Fallback: SIFT 失败场景，用全局中位 H
        warnings.warn(f"PanAligner parse failed for {reg_txt_path}: {e}. "
                      "Falling back to median H matrix.", UserWarning, stacklevel=2)
        with open(MEDIAN_H_PATH) as f:
            fb = json.load(f)
        H_mat = np.array(fb["H"], dtype=np.float64)
        x0, y0 = fb["crop"][0], fb["crop"][1]
        ys, xs = np.mgrid[0:H, 0:W]
        flat = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
        mapped = H_mat @ flat
        mapped = mapped[:2] / mapped[2]
        PX = mapped[0].reshape(H, W) - x0
        PY = mapped[1].reshape(H, W) - y0

    return cv2.remap(
        pan.astype(np.float32),
        PX.astype(np.float32), PY.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def load_6channel(stem, category):
    """加载 6 通道张量。Returns: (6, H, W) float32 [b1,b2,b3,b4,b5,pan]"""
    cal_dir = CALIBRATED_DIR / category
    bands = []
    for i in range(1, 6):
        b = tifffile.imread(str(cal_dir / f"{stem}_b{i}.tif"))
        bands.append(b.astype(np.float32))
    pan = resample_pan_to_b2(
        cal_dir / f"{stem}_pan.tif",
        cal_dir / f"{stem}_pan_reg.txt",
    )
    bands.append(pan)
    return np.stack(bands, axis=0)


def normalize_tensor(tensor, norm_stats):
    """Per-band z-score 归一化。tensor: (6,H,W) or (B,6,H,W)"""
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    if tensor.ndim == 3:
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif tensor.ndim == 4:
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
    return (tensor - mean) / np.maximum(std, 1e-6)


def load_normalized(stem, category, norm_stats):
    """加载 + 归一化。Returns: (6, H, W) float32 z-scored"""
    return normalize_tensor(load_6channel(stem, category), norm_stats)


def load_norm_stats():
    """加载归一化统计文件"""
    with open(NORM_STATS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 基线模型: 仅 pan 输入 (对比实验用)
# ---------------------------------------------------------------------------
def load_pan_only(stem, category):
    """加载仅 pan 通道(重采样到 b2 网格). Returns: (1, H, W) float32.

    用于基线模型对比实验: 与 load_6channel 唯一区别是只取 pan 通道,
    其余(配准/重采样/fallback)完全一致, 保证对比公平.
    """
    cal_dir = CALIBRATED_DIR / category
    pan = resample_pan_to_b2(
        cal_dir / f"{stem}_pan.tif",
        cal_dir / f"{stem}_pan_reg.txt",
    )
    return pan[None, :, :]  # (1, H, W)


def normalize_pan_tensor(tensor, norm_stats_pan):
    """Per-band z-score 归一化(仅 pan 单通道).

    Args:
        tensor: (1, H, W) or (B, 1, H, W) float32
        norm_stats_pan: dict with "mean": [m], "std": [s]  (pan only)
    """
    mean = np.array(norm_stats_pan["mean"], dtype=np.float32)
    std = np.array(norm_stats_pan["std"], dtype=np.float32)
    if tensor.ndim == 3:
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif tensor.ndim == 4:
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
    return (tensor - mean) / np.maximum(std, 1e-6)


def load_norm_stats_pan():
    """加载 pan-only 归一化统计文件."""
    pan_stats_path = CONFIGS_DIR / "norm_stats_pan.json"
    with open(pan_stats_path) as f:
        return json.load(f)


if __name__ == "__main__":
    from config import print_paths
    print_paths()
    stats = load_norm_stats()
    print(f"Norm stats loaded: {len(stats['scenes'])} scenes, bands={stats['band_names']}")
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")
    # 基线 pan-only
    try:
        pan_stats = load_norm_stats_pan()
        print(f"\nPan-only stats: {pan_stats.get('n_scenes', '?')} scenes")
        print(f"  Mean: {pan_stats['mean']}")
        print(f"  Std:  {pan_stats['std']}")
    except FileNotFoundError:
        print("\nPan-only stats not found. Run: python recompute_norm_stats_pan.py")
