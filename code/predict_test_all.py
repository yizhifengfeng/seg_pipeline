# -*- coding: utf-8 -*-
"""
测试集全量批量预测对比（三模型横向对比）
==========================================
用 outputV4 下三个训练目录的最优模型 (best_model.pth, 已含 EMA 平滑权重)
分别对 split.json 的 test 集 (34 个场景) 做整图滑窗预测, 预测结果按模型
分文件夹存放, 并输出定量指标对比与可视化对比图。

用法 (云端 AutoDL, 与训练相同的环境变量):
    python predict_test_all.py
    python predict_test_all.py --models-root /root/autodl-tmp/outputV4
    python predict_test_all.py --tta --conf-threshold 0.5   # 可选增强

本地 (Windows) 示例:
    set MS_CRACKSEG_DATA=D:/deeplearning/MS-CrackSeg-main/lab
    python predict_test_all.py --models-root ../outputV4

输出结构:
    <output-root>/
        train/               # 多光谱 6ch DeepLabV3+ (CBAM+Sobel)
            {stem}_pred.png        # 彩色预测掩膜
            {stem}_overlay.png     # RGB 50% 叠加
            metrics.json           # 测试集聚合指标
            confusion_matrix.png
        train_baseline/      # pan-only 基线
            ...
        train_intermediate/  # 中间融合 (注意: 该模型仅训练 42 epoch, 未收敛)
            ...
        comparison/          # 三模型并排对比 + 汇总
            {stem}_compare.png     # RGB | GT | 三模型预测 五联图
            metrics_comparison.json
            metrics_comparison.csv
            per_class_iou_bars.png
            summary_table.md

推理协议与训练期 validate()/evaluate_test() 完全一致:
    滑窗 crop=512, stride=256, 无 TTA, argmax, 指标跳过 ignore(255) 像素。
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

# ── 先解析参数再导入 config (允许 --data-root / --root 覆盖环境) ──
parser = argparse.ArgumentParser(description="三模型测试集全量预测对比")
parser.add_argument("--models-root", default=None,
                    help="包含 train/ train_baseline/ train_intermediate/ 的目录 "
                         "(默认自动探测 outputs/ 或 outputV4/)")
parser.add_argument("--output-root", default=None,
                    help="预测结果输出目录 (默认 <models-root>/test_predictions)")
parser.add_argument("--data-root", default=None,
                    help="覆盖数据根目录 (含 calibrated/), 等效 MS_CRACKSEG_DATA")
parser.add_argument("--root", default=None,
                    help="覆盖工程根目录 (含 configs/ data/masks/), 等效 MS_CRACKSEG_ROOT")
parser.add_argument("--split", default="test", choices=["test", "val"],
                    help="预测哪个划分 (默认 test)")
parser.add_argument("--tta", action="store_true",
                    help="启用水平翻转 TTA (默认关闭, 与训练期验证协议一致)")
parser.add_argument("--conf-threshold", type=float, default=0.0,
                    help="置信度阈值, 低于此值非背景像素归背景 (0=关闭, 默认关闭)")
parser.add_argument("--models", nargs="+",
                    default=["train", "train_baseline", "train_intermediate"],
                    help="要评测的模型目录名 (默认全部三个)")
parser.add_argument("--save-compare", action="store_true", default=True)
args = parser.parse_args()

if args.data_root:
    import os
    os.environ["MS_CRACKSEG_DATA"] = str(Path(args.data_root).resolve())
if args.root:
    import os
    os.environ["MS_CRACKSEG_ROOT"] = str(Path(args.root).resolve())

import torch

from config import (ROOT, MASKS_DIR, CONFIGS_DIR, CLASS_NAMES, CLASS_COLORS_BGR,
                    CLASS_COLORS_RGB, H, W, IGNORE_INDEX, NUM_CLASSES, OUTPUT_DIR)
from build_input_tensor import (load_6channel, normalize_tensor,
                                load_pan_only, normalize_pan_tensor)
from predict_utils import sliding_window_predict
from model import DeepLabV3Plus, DeepLabV3PlusBaseline

# 中间融合模型类: 独立文件加载, 避免与 code/model.py 类名冲突
_FUSION_PY = ROOT / "INTERMEDIATE_FUSION" / "model.py"
if not _FUSION_PY.exists():
    _FUSION_PY = CODE_DIR.parent / "INTERMEDIATE_FUSION" / "model.py"
_spec = importlib.util.spec_from_file_location("intermediate_model", str(_FUSION_PY))
_imod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_imod)
DeepLabV3PlusIntermediateFusion = _imod.DeepLabV3PlusIntermediateFusion


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def load_norm_stats(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_state(model_path, device):
    """best_model.pth 统一为 {"model": ...}; 兼容旧键名。"""
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    if "model" in ckpt:
        return ckpt["model"], ckpt
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    raise KeyError(f"{model_path} 中未找到 model 权重, keys={list(ckpt.keys())[:8]}")


def colorize(mask, ignore_value=None):
    """类别索引 -> BGR 彩色图; 可选把 ignore 像素画成灰色。"""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, bgr in CLASS_COLORS_BGR.items():
        color[mask == idx] = bgr
    if ignore_value is not None:
        color[mask == IGNORE_INDEX] = (128, 128, 128)
    return color


def rgb_preview(tensor):
    """(6,H,W) 反射率张量 -> BGR uint8 预览 (b3,b2,b1 百分位拉伸)。"""
    rgb = np.stack([tensor[2], tensor[1], tensor[0]], axis=-1)
    p2, p98 = np.percentile(rgb, [2, 98])
    out = (np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class MetricTracker:
    """与 train.py.MetricTracker 一致的混淆矩阵指标 (ignore_index 支持)。"""

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
        idx = target[valid] * self.nc + pred[valid]
        bins = np.bincount(idx, minlength=self.nc * self.nc)
        self.confusion += bins.reshape(self.nc, self.nc)

    def compute(self):
        c = self.confusion
        tp = np.diag(c)
        fp = c.sum(0) - tp
        fn = c.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-6)
        return {
            "mIoU": float(np.nanmean(iou)),
            "OA": float(tp.sum() / max(c.sum(), 1)),
            "per_class_IoU": {CLASS_NAMES[i]: float(iou[i]) for i in range(self.nc)},
            "per_class_F1": {CLASS_NAMES[i]: float(f1[i]) for i in range(self.nc)},
            "per_class_Precision": {CLASS_NAMES[i]: float(prec[i]) for i in range(self.nc)},
            "per_class_Recall": {CLASS_NAMES[i]: float(rec[i]) for i in range(self.nc)},
        }


def balance_score(m):
    """与 train.py.compute_balance_score 相同的综合评分。"""
    defect = CLASS_NAMES[1:]
    f1 = np.mean([m["per_class_F1"][c] for c in defect])
    p = np.mean([m["per_class_Precision"][c] for c in defect])
    r = np.mean([m["per_class_Recall"][c] for c in defect])
    return 0.4 * m["mIoU"] + 0.3 * f1 + 0.15 * p + 0.15 * r


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    cmf = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmf, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cmf[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmf[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ---------------------------------------------------------------------------
# 模型规格
# ---------------------------------------------------------------------------
def detect_models_root():
    candidates = []
    if args.models_root:
        candidates.append(Path(args.models_root))
    candidates += [OUTPUT_DIR, ROOT / "outputV4", CODE_DIR.parent / "outputV4"]
    for c in candidates:
        if (c / args.models[0] / "best_model.pth").exists():
            return c.resolve()
    raise FileNotFoundError(
        "未找到模型目录, 请用 --models-root 指定包含 "
        f"{args.models} 的目录。尝试过: {[str(c) for c in candidates]}")


MODELS_ROOT = detect_models_root()
OUT_ROOT = Path(args.output_root).resolve() if args.output_root \
    else MODELS_ROOT / "test_predictions"


def build_model(run_name, device):
    """按训练目录加载对应结构与最优权重。"""
    run_dir = MODELS_ROOT / run_name
    model_path = run_dir / "best_model.pth"
    state, ckpt = load_checkpoint_state(model_path, device)

    if run_name == "train_baseline":
        model = DeepLabV3PlusBaseline(num_classes=NUM_CLASSES,
                                      backbone_pretrained=False, output_stride=16)
        input_mode = "pan"
        stats = load_norm_stats(CONFIGS_DIR / "norm_stats_pan.json")
    elif run_name == "train_intermediate":
        sws = float(ckpt.get("spec_width_scale", 1.0))
        model = DeepLabV3PlusIntermediateFusion(num_classes=NUM_CLASSES,
                                                backbone_pretrained=False,
                                                output_stride=16,
                                                spec_width_scale=sws)
        input_mode = "6ch"
        stats = load_norm_stats(CONFIGS_DIR / "norm_stats.json")
    else:  # train: 多光谱 6ch 早期融合
        model = DeepLabV3Plus(num_classes=NUM_CLASSES,
                              backbone_pretrained=False, output_stride=16)
        input_mode = "6ch"
        stats = load_norm_stats(CONFIGS_DIR / "norm_stats.json")

    model.load_state_dict(state)
    model.to(device).eval()
    info = {"epoch": ckpt.get("epoch"), "train_best_miou": ckpt.get("miou"),
            "input_mode": ckpt.get("input_mode"), "params_M":
            sum(p.numel() for p in model.parameters()) / 1e6}
    return model, input_mode, stats, info


@torch.no_grad()
def predict_one(model, input_mode, stats, stem, category, device,
                tta=False, conf_threshold=0.0):
    """单场景整图滑窗预测, 返回 (pred_mask, tensor6ch, prob_map)。"""
    tensor = load_6channel(stem, category)  # (6,H,W) 同时用于预览
    if input_mode == "pan":
        pan = load_pan_only(stem, category)          # (1,H,W)
        image = normalize_pan_tensor(pan, stats)
    else:
        image = normalize_tensor(tensor, stats)      # (6,H,W)
    image_t = torch.from_numpy(np.ascontiguousarray(image)).float()

    prob_map = sliding_window_predict(model, image_t, device,
                                      crop=512, stride=256, tta=tta)
    prob_np = prob_map.numpy()
    pred = prob_np.argmax(axis=0)
    if conf_threshold > 0:
        conf = prob_np.max(axis=0)
        pred[(conf < conf_threshold) & (pred != 0)] = 0
    return pred.astype(np.uint8), tensor, prob_np


def save_compare_figure(tensor, gt, preds_by_model, stem, category, out_path):
    """五联图: RGB | GT | 各模型预测。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 2 + len(preds_by_model)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.8))
    rgb = np.stack([tensor[2], tensor[1], tensor[0]], axis=-1)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1)
    axes[0].imshow(rgb); axes[0].set_title("RGB", fontsize=9)

    def panel(ax, msk, title):
        mc = np.zeros((*msk.shape, 3), dtype=np.uint8)
        for ci, col in CLASS_COLORS_RGB.items():
            mc[msk == ci] = col
        mc[msk == IGNORE_INDEX] = (200, 200, 200)
        ax.imshow(mc); ax.set_title(title, fontsize=9)

    panel(axes[1], gt, "GT")
    for ax, (name, pred) in zip(axes[2:], preds_by_model.items()):
        panel(ax, pred, name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{category}/{stem}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (CONFIGS_DIR / "split.json").exists():
        sys.exit(f"缺少 {CONFIGS_DIR/'split.json'}")
    with open(CONFIGS_DIR / "split.json", encoding="utf-8") as f:
        split = json.load(f)
    samples = split[args.split]
    print(f"设备: {device} | 模型根: {MODELS_ROOT}")
    print(f"输出根: {OUT_ROOT}")
    print(f"{args.split} 集: {len(samples)} 个场景 "
          f"| TTA={args.tta} | conf_thr={args.conf_threshold}")

    # 数据存在性检查
    if not MASKS_DIR.exists():
        sys.exit(f"缺少掩膜目录 {MASKS_DIR} (检查 MS_CRACKSEG_ROOT)")
    from config import CALIBRATED_DIR
    if not CALIBRATED_DIR.exists():
        sys.exit(f"缺少 calibrated 数据目录 {CALIBRATED_DIR} "
                 f"(用 --data-root 指定, 或设 MS_CRACKSEG_DATA)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    per_image_rows = []   # CSV: 每图每模型各类像素数

    for run_name in args.models:
        t0 = time.time()
        model, input_mode, stats, info = build_model(run_name, device)
        out_dir = OUT_ROOT / run_name
        cmp_dir = OUT_ROOT / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmp_dir.mkdir(parents=True, exist_ok=True)

        note = ""
        if run_name == "train_intermediate":
            note = " (注意: 该模型仅训练 42 epoch, 未完成 120 epoch 调度, 结果仅供参考)"
        print(f"\n=== {run_name} | 输入={input_mode} | "
              f"参数={info['params_M']:.1f}M | ckpt epoch={info['epoch']}{note}")

        tracker = MetricTracker(NUM_CLASSES, IGNORE_INDEX)
        n_done = 0
        for i, s in enumerate(samples, 1):
            stem, category = s["stem"], s["category"]
            try:
                pred, tensor, _ = predict_one(
                    model, input_mode, stats, stem, category, device,
                    tta=args.tta, conf_threshold=args.conf_threshold)
            except FileNotFoundError as e:
                print(f"  [{i}/{len(samples)}] 跳过 {category}/{stem}: {e}")
                continue

            mask_path = MASKS_DIR / category / f"{stem}_mask.png"
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                gt = np.zeros((H, W), dtype=np.uint8)
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            tracker.update(pred, gt)

            # 保存预测与叠加
            cv2.imwrite(str(out_dir / f"{stem}_pred.png"), colorize(pred, ignore_value=IGNORE_INDEX))
            ov = cv2.addWeighted(rgb_preview(tensor), 0.5,
                                 colorize(pred, ignore_value=IGNORE_INDEX), 0.5, 0)
            cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), ov)

            row = {"image": f"{category}/{stem}", "model": run_name}
            for ci, name in enumerate(CLASS_NAMES):
                row[f"{name}_px"] = int((pred == ci).sum())
            per_image_rows.append(row)
            n_done += 1
            if i % 5 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}] {time.time()-t0:.0f}s")

        m = tracker.compute()
        m["balance_score"] = balance_score(m)
        m["model_info"] = info
        m["inference"] = {"protocol": "sliding_window crop=512 stride=256",
                          "tta": args.tta, "conf_threshold": args.conf_threshold,
                          "n_images": n_done}
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        plot_confusion(tracker.confusion, out_dir / "confusion_matrix.png",
                       f"Test Confusion - {run_name}")
        all_metrics[run_name] = m
        print(f"  -> mIoU={m['mIoU']:.4f} OA={m['OA']:.4f} "
              f"score={m['balance_score']:.4f} | 用时 {time.time()-t0:.0f}s")

        # 释放显存
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 缓存 GT/中间结果用于对比图 ──
    if args.save_compare and all_metrics:
        print("\n生成三模型并排对比图 ...")
        cmp_dir = OUT_ROOT / "comparison"
        # 重新逐图生成五联图 (复用已保存的 pred png, 无需再推理)
        for s in samples:
            stem, category = s["stem"], s["category"]
            try:
                tensor = load_6channel(stem, category)
            except FileNotFoundError:
                continue
            mask_path = MASKS_DIR / category / f"{stem}_mask.png"
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                gt = np.zeros((H, W), dtype=np.uint8)
            preds_by_model, ok = {}, True
            for run_name in all_metrics:
                p = OUT_ROOT / run_name / f"{stem}_pred.png"
                if not p.exists():
                    ok = False
                    break
                bgr = cv2.imread(str(p))
                # 灰色 ignore -> 索引 255
                idx = np.zeros(bgr.shape[:2], dtype=np.uint8)
                for ci, bgr_c in CLASS_COLORS_BGR.items():
                    idx[(bgr == np.array(bgr_c, dtype=np.uint8)).all(axis=-1)] = ci
                idx[(bgr == np.array([128, 128, 128], dtype=np.uint8)).all(axis=-1)] = IGNORE_INDEX
                preds_by_model[run_name] = idx
            if ok and preds_by_model:
                save_compare_figure(tensor, gt, preds_by_model,
                                    stem, category, cmp_dir / f"{stem}_compare.png")

        # ── 汇总输出 ──
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "model_info"}
                   for k, v in all_metrics.items()}
        summary["_model_info"] = {k: v["model_info"] for k, v in all_metrics.items()}
        with open(cmp_dir / "metrics_comparison.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # CSV
        if per_image_rows:
            with open(cmp_dir / "per_image_class_pixels.csv", "w", newline="",
                      encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                w.writeheader()
                w.writerows(per_image_rows)

        # 柱状图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        runs = list(all_metrics.keys())
        x = np.arange(len(CLASS_NAMES))
        width = 0.8 / len(runs)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for r_i, run in enumerate(runs):
            m = all_metrics[run]
            axes[0].bar(x + r_i * width,
                        [m["per_class_IoU"][c] for c in CLASS_NAMES],
                        width, label=run)
            axes[1].bar(x + r_i * width,
                        [m["per_class_F1"][c] for c in CLASS_NAMES],
                        width, label=run)
        for ax, title in zip(axes, ["Per-class IoU", "Per-class F1"]):
            ax.set_xticks(x + width * (len(runs) - 1) / 2)
            ax.set_xticklabels(CLASS_NAMES)
            ax.set_ylim(0, 1); ax.set_title(title); ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(cmp_dir / "per_class_iou_bars.png", dpi=120)
        plt.close()

        # Markdown 汇总表
        lines = ["# 三模型测试集预测对比", "",
                 f"协议: 滑窗 crop=512 stride=256 | TTA={args.tta} "
                 f"| conf_thr={args.conf_threshold} | N={len(samples)}", "",
                 "| 模型 | mIoU | OA | score | " + " | ".join(
                     f"IoU {c}" for c in CLASS_NAMES[1:]) + " |",
                 "|---|---|---|---|---|---|---|"]
        for run, m in all_metrics.items():
            lines.append(
                f"| {run} | {m['mIoU']:.4f} | {m['OA']:.4f} | "
                f"{m['balance_score']:.4f} | " + " | ".join(
                    f"{m['per_class_IoU'][c]:.4f}"
                    for c in CLASS_NAMES[1:]) + " |")
        with open(cmp_dir / "summary_table.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print("\n===== 汇总 =====")
        for run, m in all_metrics.items():
            print(f"{run:20s} mIoU={m['mIoU']:.4f} OA={m['OA']:.4f} "
                  f"score={m['balance_score']:.4f}")
        print(f"\n全部输出位于: {OUT_ROOT}")


if __name__ == "__main__":
    main()
