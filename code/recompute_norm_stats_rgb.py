# -*- coding: utf-8 -*-
"""
重算 RGB (b1/b2/b3) 归一化统计量 — 仅使用 train split
======================================================
方案A 新增 RGB 基线用: RGB 基线只用 b1/b2/b3 三个可见光波段,
需要单独的 RGB 归一化统计 (与 pan-only 统计同理).
仅在 train split 上计算, 避免数据泄露(审计 P0#5).

用法:
    python recompute_norm_stats_rgb.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SPLIT_PATH, CONFIGS_DIR, H, W
from build_input_tensor import load_rgb

RGB_BAND_NAMES = ["b1", "b2", "b3"]


def compute_rgb_norm_stats(train_scenes):
    """逐场景累加 b1/b2/b3 三通道 mean/std."""
    n_scenes = len(train_scenes)
    print(f"Computing RGB (b1/b2/b3) norm stats on {n_scenes} train scenes, "
          f"{H*W} pixels/scene")

    running_sum = np.zeros(3, dtype=np.float64)
    running_sqsum = np.zeros(3, dtype=np.float64)
    running_min = np.full(3, np.inf, dtype=np.float64)
    running_max = np.full(3, -np.inf, dtype=np.float64)
    total_pixels = 0
    errors = []
    valid_scenes = []

    for i, s in enumerate(train_scenes):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_scenes}] running mean={running_sum/total_pixels}")
        try:
            tensor = load_rgb(s["stem"], s["category"])  # (3, H, W) float32
            if tensor.shape != (3, H, W):
                print(f"  skip {s['stem']}: shape {tensor.shape}")
                errors.append({"stem": s["stem"], "reason": f"shape {tensor.shape}"})
                continue
            for c in range(3):
                running_sum[c] += tensor[c].sum()
                running_sqsum[c] += (tensor[c].astype(np.float64) ** 2).sum()
                running_min[c] = min(running_min[c], float(tensor[c].min()))
                running_max[c] = max(running_max[c], float(tensor[c].max()))
            total_pixels += H * W
            valid_scenes.append(s)
        except Exception as e:
            print(f"  skip {s['stem']}: {e}")
            errors.append({"stem": s["stem"], "reason": str(e)})

    mean = running_sum / total_pixels
    var = running_sqsum / total_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    stats = {
        "band_names": RGB_BAND_NAMES,
        "band_units": ["reflectance", "reflectance", "reflectance"],
        "band_physical_range": [{"min": running_min[c], "max": running_max[c]}
                                for c in range(3)],
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

    stats = compute_rgb_norm_stats(train_scenes)

    out_path = CONFIGS_DIR / "norm_stats_rgb.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Output: {out_path}")
    print(f"Scenes: {stats['n_scenes']} (train only)")
    print(f"Mean: {[f'{v:.6f}' for v in stats['mean']]}")
    print(f"Std:  {[f'{v:.6f}' for v in stats['std']]}")
    print(f"Physical ranges:")
    for i, name in enumerate(RGB_BAND_NAMES):
        r = stats["band_physical_range"][i]
        print(f"  {name}: [{r['min']:.4f}, {r['max']:.4f}] (reflectance)")
    print(f"Errors: {len(stats['errors'])} scenes skipped")


if __name__ == "__main__":
    main()
