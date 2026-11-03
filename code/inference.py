# -*- coding: utf-8 -*-
"""
端到端推理（云端版）
====================
python inference.py --stem img_0298 --category crack
python inference.py --stem img_0001 --category flaking --model outputs/train/best_model.pth

审计修复要点:
  - P2#16: GSD 从 config.DEFAULT_GSD_MM 读取, 仍支持 --gsd 命令行覆盖.
  - P2#17: 推理增加置信度阈值(低置信度像素归为背景) + TTA(水平翻转平均).
  - P2#12: 颜色表统一从 config.CLASS_COLORS_BGR 读取.
  - N2 修复: 推理使用滑窗 (复用 predict_utils.sliding_window_predict),
            与验证流程完全一致, 避免整图 OOM, 每个 patch 都有 autocast.
            TTA 按 patch 做 (翻转推理后翻回取平均), 不翻倍显存.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (MASKS_DIR, NORM_STATS_PATH, OUTPUT_DIR, CLASS_NAMES,
                    CLASS_COLORS_BGR, H, W, DEFAULT_GSD_MM, IGNORE_INDEX)
from build_input_tensor import load_6channel, normalize_tensor
from model import DeepLabV3Plus
from predict_utils import sliding_window_predict

INFER_DIR = OUTPUT_DIR / "inference"


def load_model(model_path, device):
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model = DeepLabV3Plus(num_classes=5, backbone_pretrained=False,
                          output_stride=16).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_full_image(model, stem, category, norm_stats, device,
                       tta=True, conf_threshold=0.5, crop=512, stride=256):
    """滑窗推理整图 + (可选)TTA + 置信度阈值 (N2 修复).

    Args:
        tta: True 时每个 patch 做水平翻转 TTA (翻转推理后翻回取平均).
        conf_threshold: softmax 最大概率低于此值的非背景像素归为背景.
        crop, stride: 滑窗参数 (与验证一致: crop=512, stride=256).
    Returns: (pred_mask, tensor, prob_map)
    """
    tensor = load_6channel(stem, category)
    image = normalize_tensor(tensor, norm_stats)
    image_t = torch.from_numpy(image).float()

    # N2 修复: 使用滑窗推理 (复用 train.py 的共享模块), 全图覆盖 + autocast + per-patch TTA
    prob_map = sliding_window_predict(
        model, image_t, device, crop=crop, stride=stride, tta=tta)

    prob_np = prob_map.numpy()
    pred = prob_np.argmax(axis=0)                   # (H, W)
    conf = prob_np.max(axis=0)                      # (H, W)

    # 置信度阈值: 低置信度的病害像素(非背景)归为背景
    if conf_threshold > 0:
        low_conf = (conf < conf_threshold) & (pred != 0)
        pred[low_conf] = 0

    return pred, tensor, prob_np


def colorize_mask(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, bgr in CLASS_COLORS_BGR.items():
        color[mask == idx] = bgr
    return color


def make_rgb_preview(tensor):
    rgb = np.stack([tensor[2], tensor[1], tensor[0]], axis=-1)
    p2, p98 = np.percentile(rgb, [2, 98])
    return (np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1) * 255).astype(np.uint8)


def compute_area_stats(mask, gsd_mm):
    """病害面积统计. 面积 = 像素数 * (GSD/10)^2 cm^2."""
    stats = {}
    total = mask.size
    for i, name in enumerate(CLASS_NAMES):
        n = int(np.sum(mask == i))
        stats[name] = {"pixels": n, "ratio_pct": round(n/total*100, 2),
                       "area_cm2": round(n * (gsd_mm/10)**2, 1)}
    return stats


def run_inference(stem, category, model_path, norm_stats, device, out_dir,
                  gsd_mm=DEFAULT_GSD_MM, tta=True, conf_threshold=0.5):
    model, ckpt = load_model(model_path, device)
    pred, tensor, prob_map = predict_full_image(
        model, stem, category, norm_stats, device,
        tta=tta, conf_threshold=conf_threshold)
    color = colorize_mask(pred)
    rgb_bgr = cv2.cvtColor(make_rgb_preview(tensor), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(rgb_bgr, 0.5, color, 0.5, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_pred.png"), color)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
    np.save(str(out_dir / f"{stem}_probmap.npy"), prob_map)
    stats = compute_area_stats(pred, gsd_mm)
    with open(out_dir / f"{stem}_stats.json", "w", encoding="utf-8") as f:
        json.dump({"stem": stem, "category": category, "GSD_mm": gsd_mm,
                   "tta": tta, "conf_threshold": conf_threshold,
                   "sliding_window": {"crop": 512, "stride": 256},
                   "class_stats": stats}, f, ensure_ascii=False, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="病害分割推理")
    parser.add_argument("--stem", required=True, help="场景ID, 如 img_0298")
    parser.add_argument("--category", required=True, help="类别: crack/flaking/biological_attachment")
    parser.add_argument("--model", default=str(OUTPUT_DIR / "train" / "best_model.pth"))
    parser.add_argument("--gsd", type=float, default=DEFAULT_GSD_MM,
                        help="地面采样距离 mm/px (默认 %(default)s)")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="启用水平翻转 TTA (默认开启)")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="关闭 TTA")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="置信度阈值, 低于此值的非背景像素归为背景 (0=关闭)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    print(f"场景: {args.category}/{args.stem} | 设备: {device} | "
          f"GSD: {args.gsd}mm/px | TTA: {args.tta} | conf>={args.conf_threshold}")
    print(f"推理方式: 滑窗 crop=512 stride=256 (与验证一致)")
    stats = run_inference(args.stem, args.category, args.model, norm_stats,
                          device, INFER_DIR, gsd_mm=args.gsd,
                          tta=args.tta, conf_threshold=args.conf_threshold)

    print(f"\n输出: {INFER_DIR}/{args.stem}_pred.png, _overlay.png, _stats.json, _probmap.npy")
    print(f"\n病害面积统计:")
    print(f"{'类别':25s} {'像素':>10s} {'占比':>8s} {'面积':>12s}")
    for cls, s in stats.items():
        print(f"{cls:25s} {s['pixels']:10d} {s['ratio_pct']:7.2f}% {s['area_cm2']:10.1f} cm^2")


if __name__ == "__main__":
    main()
