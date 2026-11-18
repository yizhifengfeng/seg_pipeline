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
# ---------------------------------------------------------------------------
# 基线模型: 仅 pan 输入 (对比实验用)
# ---------------------------------------------------------------------------
# PanAligner fallback 警告去重: 同一场景只警告一次(训练循环中同一文件每个
# epoch 都会触发, 原实现导致警告刷屏).
_WARNED_SCENES = set()


def resample_pan_to_b2(pan_path, reg_txt_path):
    """把 pan 重采样到 b2 网格 (1456x1088)。fallback 场景用中位 H。

    审计修复:
      - P2#15: bare except (ValueError, Exception) 收窄为 (ValueError, KeyError, OSError),
               并打印 warning log, 不再静默吞掉 FileNotFoundError 等真实错误.
      - P2#18: cv2.remap 边界填充从 BORDER_CONSTANT(0) 改为 BORDER_REFLECT,
               避免硬黑边被模型当作伪特征.
    v2 修复: fallback 警告按场景去重(每个 reg_txt_path 只警告一次),
             避免训练时警告刷屏.
    """
    import warnings
    pan = tifffile.imread(str(pan_path))  # uint16
    try:
        pa = PanAligner(reg_txt_path)
        PX, PY = pa.b2_grid_to_pan_out()
    except (ValueError, KeyError, OSError) as e:
        # Fallback: SIFT 失败场景，用全局中位 H (每场景只警告一次)
        if str(reg_txt_path) not in _WARNED_SCENES:
            _WARNED_SCENES.add(str(reg_txt_path))
            warnings.warn(f"PanAligner parse failed for {reg_txt_path}: {e}. "
                          "Falling back to median H matrix. "
                          f"(total fallback scenes so far: {len(_WARNED_SCENES)})",
                          UserWarning, stacklevel=2)
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


def load_rgb(stem, category):
    """加载 RGB 三通道张量 (b1/b2/b3)。Returns: (3, H, W) float32 [b1,b2,b3].

    方案A 新增 RGB 基线用: b1/b2/b3 在 calibrated 中已对齐到 b2 网格
    (1456x1088), 直接 imread 即可, 无需 pan 对齐, 零配准风险。
    物理量纲: 反射率 [0,1] (b1-b5 均做了反射率转换, 与 pan 的 DN 不同)。
    """
    cal_dir = CALIBRATED_DIR / category
    bands = []
    for i in (1, 2, 3):
        b = tifffile.imread(str(cal_dir / f"{stem}_b{i}.tif"))
        bands.append(b.astype(np.float32))
    return np.stack(bands, axis=0)


def normalize_tensor(tensor, norm_stats):
    """Per-band z-score 归一化。tensor: (6,H,W) or (B,6,H,W)

    v5.4a: 输出钳位到 ±15 —— 正常 z-score (含 pan DN 域与指数通道最坏情形)
    不超过 ±6~±12, 超界值即数据异常, 会造成 fp16 前向溢出 → BN running
    stats 污染 → eval 全背景坍缩 (云端 E2 ep2 事故). 钳位是源头防御.
    """
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    if tensor.ndim == 3:
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif tensor.ndim == 4:
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
    out = (tensor - mean) / np.maximum(std, 1e-6)
    return np.clip(out, -15.0, 15.0, out=out)


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


# ---------------------------------------------------------------------------
# 光谱指数通道 (阶段二新增, E2/E3 实验)
# ---------------------------------------------------------------------------
# 4 个精选归一化差值指数 (b1-b5 为 0 基索引), 固定顺序, 全链路唯一来源.
# 挑选依据见 lab/spectral_indices_all/phase1/ (全像素 BD + AUC + Otsu 验证):
#   ndi_b4b5: bio 可分性最强 (AUC 0.884, Otsu 零学习 IoU 0.301)
#   ndi_b2b3: flaking 空间完整性 (像素级最弱, 交模型验证)
#   ndi_b2b4: crack vs bg 全像素 BD 第一 (0.099), bio 反方向互补 (与 b4b5 r=-0.55)
#   ndi_b2b5: 整体缺陷对比度增强
# 已剔除 ndi_b1b5: 阶段一相关矩阵显示与 ndi_b2b5 r=0.71 / ndi_b2b4 r=0.61,
#   三者同源冗余, 且无独立类别贡献 (所有 AUC 均被 b2b5 覆盖).
SPECTRAL_INDICES = [
    ("ndi_b4b5", 3, 4),   # (b4-b5)/(b4+b5)
    ("ndi_b2b3", 1, 2),   # (b2-b3)/(b2+b3)
    ("ndi_b2b4", 1, 3),   # (b2-b4)/(b2+b4)
    ("ndi_b2b5", 1, 4),   # (b2-b5)/(b2+b5)
]
INDEX_NAMES = [n for n, _, _ in SPECTRAL_INDICES]
INDEX_EPS = 1e-6


def compute_index_channels(bands5):
    """从 5 波段反射率计算 4 个光谱指数通道.

    Args:
        bands5: (5, H, W) float32 [b1..b5] 反射率域 (增强后的波段同样适用;
                必须在几何/辐射增强完成后调用 —— 线性插值不保持 (a-b)/(a+b),
                预计算指数再旋转会污染数值, 见 HANDOFF 坑列表 #17 同类).
    Returns:
        (4, H, W) float32, 顺序同 SPECTRAL_INDICES.
    """
    assert bands5.shape[0] == 5, f"expect (5,H,W), got {bands5.shape}"
    out = np.empty((len(SPECTRAL_INDICES), bands5.shape[1], bands5.shape[2]),
                   dtype=np.float32)
    for k, (_, i, j) in enumerate(SPECTRAL_INDICES):
        a = bands5[i]
        b = bands5[j]
        out[k] = (a - b) / (a + b + INDEX_EPS)
    return out


def load_5band(stem, category):
    """加载 5 波段反射率 (无 pan). Returns: (5, H, W) float32 [b1..b5].

    E3 (RGB+指数) 用: 只需 b1-b3 作为模型基座输入, 但指数计算需要全部 5 波段.
    """
    cal_dir = CALIBRATED_DIR / category
    bands = []
    for i in range(1, 6):
        b = tifffile.imread(str(cal_dir / f"{stem}_b{i}.tif"))
        bands.append(b.astype(np.float32))
    return np.stack(bands, axis=0)


def load_indices_input(stem, category, mode):
    """加载 E2/E3 的完整输入张量 (未增强, val/test 与统计计算共用).

    Args:
        mode: "ms"  -> (10, H, W) [b1..b5, pan, ndi x4]
              "rgb" -> (7, H, W)  [b1, b2, b3, ndi x4]
    """
    if mode == "ms":
        base = load_6channel(stem, category)          # (6, H, W)
        idx = compute_index_channels(base[0:5])      # 从原始反射率计算
        return np.concatenate([base, idx], axis=0)
    elif mode == "rgb":
        bands5 = load_5band(stem, category)          # (5, H, W)
        idx = compute_index_channels(bands5)
        return np.concatenate([bands5[0:3], idx], axis=0)
    raise ValueError(f"unknown mode: {mode}")


def load_norm_stats_indices(mode):
    """加载 E2/E3 归一化统计文件 (train-only).

        mode="ms"  -> norm_stats_indices.json       (10 通道)
        mode="rgb" -> norm_stats_rgb_indices.json   (7 通道)
    """
    fname = {"ms": "norm_stats_indices.json",
             "rgb": "norm_stats_rgb_indices.json"}[mode]
    path = CONFIGS_DIR / fname
    with open(path) as f:
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
