# -*- coding: utf-8 -*-
"""
重算 E2/E3 光谱指数增强模型的归一化统计量 — 仅使用 train split
================================================================
阶段二新增. 一次遍历同时产出两个文件:
  - norm_stats_indices.json      (E2: ms 模式, 10 通道 [b1..b5, pan, ndi x4])
  - norm_stats_rgb_indices.json  (E3: rgb 模式, 7 通道 [b1,b2,b3, ndi x4])

指数统计从原始 (未增强) 反射率计算, 与 Dataset 中 val/test 路径的
compute_index_channels 完全同源 (增强会扰动分布, 但统计口径与现有
norm_stats.json 的做法一致: 均在未增强数据上计算).

仅在 train split 上计算, 避免数据泄露 (审计 P0#5, 坑 #6/#17 同理).

用法:
    python recompute_norm_stats_indices.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SPLIT_PATH, CONFIGS_DIR, H, W
from build_input_tensor import load_6channel, compute_index_channels, INDEX_NAMES

MS_BAND_NAMES = ["b1", "b2", "b3", "b4", "b5", "pan"] + INDEX_NAMES       # 10ch
RGB_BAND_NAMES = ["b1", "b2", "b3"] + INDEX_NAMES                          # 7ch


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_scenes = split["train"]
    n_scenes = len(train_scenes)
    print(f"Train split: {n_scenes} scenes (train only, no leakage)")

    # E2: 10ch 累加器; E3: 7ch 累加器 (通道数从 BAND_NAMES 派生, 增删指数自动同步)
    n_ms, n_rgb = len(MS_BAND_NAMES), len(RGB_BAND_NAMES)
    ms_sum = np.zeros(n_ms, dtype=np.float64)
    ms_sq = np.zeros(n_ms, dtype=np.float64)
    rgb_sum = np.zeros(n_rgb, dtype=np.float64)
    rgb_sq = np.zeros(n_rgb, dtype=np.float64)
    total_pixels = 0
    errors, valid_scenes = [], []

    for i, s in enumerate(train_scenes):
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{n_scenes}] ...")
        try:
            base = load_6channel(s["stem"], s["category"])          # (6, H, W)
            if base.shape != (6, H, W):
                errors.append({"stem": s["stem"], "reason": f"shape {base.shape}"})
                continue
            idx = compute_index_channels(base[0:5])                  # (4, H, W)
            ms_stack = np.concatenate([base, idx], axis=0)          # (10, H, W)
            rgb_stack = np.concatenate([base[0:3], idx], axis=0)     # (7, H, W)

            for c in range(n_ms):
                x = ms_stack[c].astype(np.float64)
                ms_sum[c] += x.sum(); ms_sq[c] += (x * x).sum()
            for c in range(n_rgb):
                x = rgb_stack[c].astype(np.float64)
                rgb_sum[c] += x.sum(); rgb_sq[c] += (x * x).sum()
            total_pixels += H * W
            valid_scenes.append(s)
        except Exception as e:
            print(f"  skip {s['stem']}: {e}")
            errors.append({"stem": s["stem"], "reason": str(e)})

    def finalize(sums, sqs, band_names):
        mean = sums / total_pixels
        var = sqs / total_pixels - mean ** 2
        std = np.sqrt(np.maximum(var, 0))
        return {
            "band_names": band_names,
            "n_scenes": len(valid_scenes),
            "n_pixels_per_band": total_pixels,
            "split": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "scenes": valid_scenes,
            "errors": errors,
        }

    ms_stats = finalize(ms_sum, ms_sq, MS_BAND_NAMES)
    rgb_stats = finalize(rgb_sum, rgb_sq, RGB_BAND_NAMES)

    for fname, stats in [("norm_stats_indices.json", ms_stats),
                         ("norm_stats_rgb_indices.json", rgb_stats)]:
        out_path = CONFIGS_DIR / fname
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nOutput: {out_path}")
        print(f"  channels: {len(stats['band_names'])} | scenes: {stats['n_scenes']}")
        print(f"  mean: {[f'{v:.4f}' for v in stats['mean']]}")
        print(f"  std:  {[f'{v:.4f}' for v in stats['std']]}")

    print(f"\nErrors: {len(errors)} scenes skipped")
    if errors:
        for e in errors[:10]:
            print(f"  {e['stem']}: {e['reason']}")


if __name__ == "__main__":
    main()
