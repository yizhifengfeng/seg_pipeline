# -*- coding: utf-8 -*-
"""
重算 pan-only 归一化统计量 — 仅使用 train split
==================================================
对比实验用: 基线模型只用 pan 通道, 需要单独的 pan 归一化统计.
仅在 train split 上计算, 避免数据泄露(审计 P0#5).

用法:
    python recompute_norm_stats_pan.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SPLIT_PATH, CONFIGS_DIR, H, W
from build_input_tensor import load_pan_only


def compute_pan_norm_stats(train_scenes):
    """逐场景累加 pan 单通道 mean/std."""
    n_scenes = len(train_scenes)
    print(f"Computing pan-only norm stats on {n_scenes} train scenes, "
          f"{H*W} pixels/scene")

    running_sum = np.zeros(1, dtype=np.float64)
    running_sqsum = np.zeros(1, dtype=np.float64)
    running_min = np.full(1, np.inf, dtype=np.float64)
    running_max = np.full(1, -np.inf, dtype=np.float64)
    total_pixels = 0
    errors = []
    valid_scenes = []

    for i, s in enumerate(train_scenes):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_scenes}] running mean={running_sum/total_pixels}")
        try:
            tensor = load_pan_only(s["stem"], s["category"])  # (1, H, W) float32
            if tensor.shape != (1, H, W):
                print(f"  skip {s['stem']}: shape {tensor.shape}")
                errors.append({"stem": s["stem"], "reason": f"shape {tensor.shape}"})
                continue
            running_sum[0] += tensor[0].sum()
            running_sqsum[0] += (tensor[0].astype(np.float64) ** 2).sum()
            running_min[0] = min(running_min[0], float(tensor[0].min()))
            running_max[0] = max(running_max[0], float(tensor[0].max()))
            total_pixels += H * W
            valid_scenes.append(s)
        except Exception as e:
            print(f"  skip {s['stem']}: {e}")
            errors.append({"stem": s["stem"], "reason": str(e)})

    mean = running_sum / total_pixels
    var = running_sqsum / total_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    stats = {
        "band_names": ["pan"],
        "band_units": ["DN (uint16)"],
        "band_physical_range": [{"min": running_min[0], "max": running_max[0]}],
        "n_scenes": len(valid_scenes),
        "n_pixels_per_band": total_pixels,
        "split": "train",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scenes": valid_scenes,
        "errors": errors,
    }
    return stats


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_scenes = split["train"]
    print(f"Train split: {len(train_scenes)} scenes")

    stats = compute_pan_norm_stats(train_scenes)

    out_path = CONFIGS_DIR / "norm_stats_pan.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Output: {out_path}")
    print(f"Scenes: {stats['n_scenes']} (train only)")
    print(f"Mean: [{stats['mean'][0]:.6f}]")
    print(f"Std:  [{stats['std'][0]:.6f}]")
    print(f"Physical range: [{stats['band_physical_range'][0]['min']:.4f}, "
          f"{stats['band_physical_range'][0]['max']:.4f}] (DN uint16)")
    print(f"Errors: {len(stats['errors'])} scenes skipped")


if __name__ == "__main__":
    main()
