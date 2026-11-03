# -*- coding: utf-8 -*-
"""
重算归一化统计量 — 仅使用 train split (审计 P0#5: 修复数据泄露)
================================================================
原 norm_stats.json 在全部 334 个场景(train+val+test)上计算, 导致 val/test
场景的分布信息泄露到归一化参数中. 本脚本仅在 train split 的 233 个场景上
重算 mean/std, 输出到 configs/norm_stats_train.json.

用法:
    python recompute_norm_stats.py          # 默认输出 norm_stats_train.json
    python recompute_norm_stats.py --overwrite  # 覆盖原 norm_stats.json
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CALIBRATED_DIR, SPLIT_PATH, NORM_STATS_PATH, CONFIGS_DIR,
                    H, W, BAND_NAMES)
from build_input_tensor import load_6channel


def compute_norm_stats(train_scenes):
    """逐场景累加 6 通道的 mean/std (Welford 在线算法避免溢出)."""
    n_pixels_per_band = H * W
    n_scenes = len(train_scenes)
    print(f"Computing norm stats on {n_scenes} train scenes, "
          f"{n_pixels_per_band} pixels/band/scene")

    # 用双遍法: 第一遍算 mean, 第二遍算 std (避免数值不稳定)
    # 对于 5 个反射率波段 [0,1] 和 pan [0, 65535], 直接累加即可
    running_sum = np.zeros(6, dtype=np.float64)
    running_sqsum = np.zeros(6, dtype=np.float64)
    running_min = np.full(6, np.inf, dtype=np.float64)
    running_max = np.full(6, -np.inf, dtype=np.float64)
    total_pixels = 0
    errors = []
    valid_scenes = []

    for i, s in enumerate(train_scenes):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_scenes}] running mean={running_sum/total_pixels}")
        try:
            tensor = load_6channel(s["stem"], s["category"])  # (6, H, W) float32
            if tensor.shape != (6, H, W):
                print(f"  skip {s['stem']}: shape {tensor.shape} != {(6, H, W)}")
                errors.append({"stem": s["stem"], "reason": f"shape {tensor.shape}"})
                continue
            # 逐通道累加 (含 min/max 用于记录物理范围, N4)
            for c in range(6):
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
    # std = sqrt(E[x^2] - E[x]^2)
    var = running_sqsum / total_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    # N4: 记录每个波段的物理范围和量纲, 便于论文解释
    # b1-b5: 定标反射率 (~0-1), pan: uint16 DN 值 (0-65535)
    band_units = ["reflectance", "reflectance", "reflectance",
                  "reflectance", "reflectance", "DN (uint16)"]

    stats = {
        "band_names": BAND_NAMES,
        "band_units": band_units,
        "band_physical_range": [{"min": running_min[c], "max": running_max[c]}
                                for c in range(6)],
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
    parser = argparse.ArgumentParser(description="重算 norm_stats (train only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖原 norm_stats.json (否则输出 norm_stats_train.json)")
    args = parser.parse_args()

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_scenes = split["train"]
    print(f"Train split: {len(train_scenes)} scenes")

    stats = compute_norm_stats(train_scenes)

    out_path = NORM_STATS_PATH if args.overwrite else CONFIGS_DIR / "norm_stats_train.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Output: {out_path}")
    print(f"Scenes: {stats['n_scenes']} (train only)")
    print(f"Mean: {[f'{v:.6f}' for v in stats['mean']]}")
    print(f"Std:  {[f'{v:.6f}' for v in stats['std']]}")
    print(f"Physical ranges (N4):")
    for i, name in enumerate(BAND_NAMES):
        r = stats["band_physical_range"][i]
        u = stats["band_units"][i]
        print(f"  {name}: [{r['min']:.4f}, {r['max']:.4f}] ({u})")
    print(f"Errors: {len(stats['errors'])} scenes skipped")
    if not args.overwrite:
        print(f"\nTo use train-only stats, update config.py or set:")
        print(f"  NORM_STATS_PATH = CONFIGS_DIR / 'norm_stats_train.json'")
        print(f"Or re-run with --overwrite to replace the original.")


if __name__ == "__main__":
    main()
