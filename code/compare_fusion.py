# -*- coding: utf-8 -*-
"""
七模型对比 + 训练效果评估 (v5.5, 2026-11-10)
=================================================
读取 outputV5/ 下各训练目录的 metrics.json + config_used.json, 生成一份
"模型训练效果" 对比报告: 精度(总体/分类别) + 收敛过程(曲线/最佳 epoch/训练预算/稳定性)
+ 增益分解.

七模型 (同一数据划分 214/60/34 seed=42, 同一损失, 同一增强; 训练预算见报告第 4 节):

  谱段基线  pan            1ch pan + 1 edge = 2ch         -> train_baseline
            rgb            3ch b1/b2/b3 + 1 edge = 4ch    -> train_rgb
  融合方式  early          5 spec + pan + 2 edge = 8ch    -> train
            intermediate   双塔 spec_width_scale=1.0       -> train_intermediate
            int_narrow     双塔 spec_width_scale=0.5       -> train_intermediate_narrow
  指数增强  ms_indices     6ch + 4 NDI + 2 edge = 12ch    -> train_indices_ms   (E2)
            rgb_indices    3ch RGB + 4 NDI + 1 edge = 8ch  -> train_indices_rgb  (E3)

产出 (默认 <outputs-root>/comparison/seven_models/):
  fusion_comparison_report.html    主报告 (表格 + 图 + 自动结论 + 公平性声明)
  overall_comparison.png           mIoU / OA / balance score, 带 pan 参考线
  per_class_comparison.png         分类别 IoU 分组柱状图
  per_class_gain_heatmap.png       分类别 IoU 相对 pan 的增益热力图 (类别级假设兑现)
  gain_decomposition.png           增益分解 (彩色/光谱/融合方式/容量/指数)
  training_curves_comparison.png   训练曲线 (val mIoU / train loss / val score, 标注最佳 epoch)
  convergence_efficiency.png       训练预算哑铃图 + best_val vs test 泛化对照
  comparison_summary.json          机器可读汇总 (数字可二次使用)

设计要点:
  - 结果目录用 --outputs-root 指定, 缺省优先 outputV5 (v5.3/v5.4 正式结果都在这里),
    没有 outputV5 才回落到 config 的 OUTPUT_DIR, 兼容 outputs/ 时代的老结果.
  - 任一模型缺失 metrics.json 自动降级为可用模型数的对比, 不报错 (HANDOFF 坑 #19:
    报告代码必须在"模型缺失"的降级路径上跑一遍).
  - 判定阈值 SIG_DELTA = 0.02 (2 个 mIoU 点). 小于阈值一律标注"无显著差异",
    单 seed 结论不做超额表述.
  - 训练预算不一致 (80 / 110 / 200 epoch) 会自动检测并写进报告第 4、7 节,
    防止把预算差异误读成结构或输入增益.

用法:
    python compare_fusion.py                          # 默认 test, 自动找 outputV5
    python compare_fusion.py --source val
    python compare_fusion.py --outputs-root ../outputV5
    python compare_fusion.py --intermediate-narrow D:/.../metrics.json
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, OUTPUT_DIR, CLASS_NAMES

# 显著性阈值: 2 个 mIoU 点 (项目既有判定惯例, 见 compare_models.py 的公平性声明)
SIG_DELTA = 0.02

DEFECT_CLASSES = ["crack", "biological_attachment", "flaking"]
CLS_SHORT = {"background": "bg", "crack": "crack",
             "biological_attachment": "bio", "flaking": "flaking"}

# ── 七模型定义 ────────────────────────────────────────────────
# short: 增益分解/图例用的短名; family: 实验分组; color: ColorBrewer Set1 (色盲友好)
MODELS = {
    "pan": dict(
        short="pan", dir="train_baseline", label="Pan 基线 (1ch)", color="#e41a1c",
        family="谱段基线", input="1ch pan + 1 edge = 2ch"),
    "rgb": dict(
        short="rgb", dir="train_rgb", label="RGB 基线 (3ch)", color="#ff7f00",
        family="谱段基线", input="3ch b1/b2/b3 + 1 edge = 4ch"),
    "early": dict(
        short="early", dir="train", label="早期融合 (6ch)", color="#377eb8",
        family="融合方式", input="5 spec + pan + 2 edge = 8ch"),
    "intermediate": dict(
        short="int-wide", dir="train_intermediate", label="中期融合 宽塔 (1.0)", color="#984ea3",
        family="融合方式", input="双塔特征级融合, spec_width_scale=1.0"),
    "intermediate_narrow": dict(
        short="int-narrow", dir="train_intermediate_narrow", label="中期融合 窄塔 (0.5)",
        color="#4d4d4d", family="融合方式",
        input="双塔特征级融合, spec_width_scale=0.5"),
    "ms_indices": dict(
        short="ms+idx", dir="train_indices_ms", label="早期+指数 (10ch)", color="#4daf4a",
        family="光谱指数", input="6ch base + 4 NDI + 2 edge = 12ch"),
    "rgb_indices": dict(
        short="rgb+idx", dir="train_indices_rgb", label="RGB+指数 (7ch)", color="#a65628",
        family="光谱指数", input="3ch RGB + 4 NDI + 1 edge = 8ch"),
}
MODEL_ORDER = ["pan", "rgb", "early", "intermediate",
               "intermediate_narrow", "ms_indices", "rgb_indices"]

# 参数量回退: metrics/config_used 未记录时使用 (数值来源见 HANDOFF / model.py 自检)
DEFAULT_PARAMS = {
    "pan": (26.75e6, "model.py 自检记录"),
    "rgb": (26.76e6, "model.py 自检记录"),
    "early": (26.77e6, "model.py 自检记录"),
    "intermediate": (48.75e6, "HANDOFF 记录"),
    "intermediate_narrow": (32.61e6, "HANDOFF 记录"),
    "ms_indices": (26.79e6, "model.py 自检记录"),
    "rgb_indices": (26.77e6, "model.py 自检记录"),
}

# ── 增益项定义 (delta = MODELS[a] − MODELS[b]) ─────────────────
GAIN_DEFS = [
    dict(key="Δ_color", short="彩色增益", a="rgb", b="pan", group="光谱 / 彩色分解",
         desc="可见光内部: 分谱(b1/b2/b3) vs 全谱灰度(pan)"),
    dict(key="Δ_spectral", short="光谱增益", a="early", b="rgb", group="光谱 / 彩色分解",
         desc="NIR(b4)/RedEdge(b5) 的纯光谱贡献"),
    dict(key="Δ_total", short="总增益", a="early", b="pan", group="光谱 / 彩色分解",
         desc="光谱 + 彩色叠加的总增益"),
    dict(key="Δ_fusion_wide", short="融合方式 宽塔", a="intermediate", b="early",
         group="融合方式", desc="特征级中期融合 vs 像素级早期融合 (48.7M vs 26.8M)"),
    dict(key="Δ_capacity", short="容量消融 窄塔", a="intermediate_narrow", b="intermediate",
         group="融合方式", desc="spec_width_scale 1.0 → 0.5, 参数 48.7M → 32.6M"),
    dict(key="Δ_fusion_narrow", short="融合方式 窄塔", a="intermediate_narrow", b="early",
         group="融合方式", desc="参数近邻对照 (32.6M vs 26.8M)"),
    dict(key="Δ_index", short="指数增益 E2", a="ms_indices", b="early", group="光谱指数",
         desc="原始 6ch 之上叠加 4 个 NDI 的额外增益"),
    dict(key="Δ_index_rgb", short="指数增益 E3", a="rgb_indices", b="rgb", group="光谱指数",
         desc="RGB 之上叠加 4 个 NDI 的增益"),
]


# ══════════════════════════════════════════════════════════════
# 基础工具
# ══════════════════════════════════════════════════════════════
def _setup_cjk_font():
    """配置 matplotlib 中文字体, 避免图表中文乱码 (Windows: YaHei / SimHei)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        for name in ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"):
            for f in font_manager.fontManager.ttflist:
                if f.name == name:
                    matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return True
    except Exception:
        pass
    return False


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [WARN] 读取失败 {p}: {exc}")
        return None


def fmt(v, prec=4, plus=False):
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if not isinstance(v, (int, float)):
        return "N/A"
    return f"{v:+.{prec}f}" if plus else f"{v:.{prec}f}"


def dspan(d, prec=4):
    """带符号与颜色的 Δ 单元格; |Δ| < SIG_DELTA 标灰 (= 无显著差异)."""
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return "N/A"
    if d > SIG_DELTA:
        color = "#1e8449"
    elif d < -SIG_DELTA:
        color = "#c0392b"
    else:
        color = "#7f8c8d"
    return f'<span style="color:{color};font-weight:600">{d:+.{prec}f}</span>'


def verdict(d):
    """把 Δ 翻译成一句判定, 严格遵守 2 点阈值."""
    if d is None:
        return "N/A"
    if abs(d) < SIG_DELTA:
        return "无显著差异"
    return "提升" if d > 0 else "下降"


def gain_phrase(d):
    """更自然的结论措辞: 未观察到显著增益 / 观察到增益 / 反而下降."""
    if d is None:
        return "无法判定"
    if abs(d) < SIG_DELTA:
        return "未观察到显著增益"
    return "观察到增益" if d > 0 else "反而下降"


def marginal_note(d):
    """Δ 刚越过阈值 (1.0–1.25 倍) 时提示属边缘差异."""
    if d is None:
        return ""
    if SIG_DELTA <= abs(d) < SIG_DELTA * 1.25:
        return "该差值刚刚越过判定阈值，属边缘差异，需重复实验支撑。"
    return ""


def rel_pct(delta, base):
    if delta is None or not base:
        return "N/A"
    return f"{delta / base * 100:+.1f}%"


# ══════════════════════════════════════════════════════════════
# 读取与指标提取
# ══════════════════════════════════════════════════════════════
def resolve_outputs_root(arg):
    """结果根目录: 显式参数 > ROOT/outputV5 > config.ROOT/outputs."""
    if arg:
        return Path(arg).expanduser(), "参数指定"
    cand = ROOT / "outputV5"
    if cand.is_dir():
        return cand, "默认 outputV5"
    return OUTPUT_DIR, "默认 config.ROOT/outputs"


def load_n_params(name, metrics_path):
    """参数量: 同目录 config_used.json 的 n_params 优先, 否则回退记录值."""
    cfg = Path(metrics_path).parent / "config_used.json"
    d = load_json(cfg)
    if d and d.get("n_params"):
        try:
            return int(d["n_params"]), "训练记录"
        except (TypeError, ValueError):
            pass
    fallback = DEFAULT_PARAMS.get(name)
    if fallback:
        return int(fallback[0]), fallback[1]
    return 0, "未知"


def extract_key_metrics(m, source="test"):
    """提取精度指标. source: 'test' | 'val'.

    兼容 final_val(新) / final_metrics(旧) / best_val 三种历史字段.
    """
    if m is None:
        return None
    used = source
    if source == "test" and m.get("test"):
        data = m["test"]
    else:
        data = m.get("final_val") or m.get("final_metrics") or m.get("best_val") or {}
        used = "val"
        if not data:
            hist = m.get("history", {})
            if hist.get("val_miou"):
                bi = int(np.argmax(hist["val_miou"]))
                data = {"mIoU": hist["val_miou"][bi], "OA": 0,
                        "per_class_IoU": {}, "per_class_F1": {},
                        "per_class_Precision": {}, "per_class_Recall": {}}
                used = "val(history best)"
    if not data:
        return None

    # balance score: 能从 per-class P/R 重算就用重算值 (与选优口径一致),
    # 否则退回 metrics.json 里记录的 best_score (最佳 epoch 的 val 选优分)
    score = m.get("best_score")
    if data.get("per_class_Precision") and data.get("per_class_Recall"):
        try:
            from train import compute_balance_score as _cbs
            score = float(_cbs(data))
        except Exception:
            pass

    pci = data.get("per_class_IoU", {}) or {}
    result = {
        "source": used,
        "mIoU": data.get("mIoU", 0),
        "OA": data.get("OA", 0),
        "score": score,
        "per_class_IoU_all": {c: pci.get(c, 0) for c in CLASS_NAMES},
        "per_class_IoU": {c: pci.get(c, 0) for c in DEFECT_CLASSES},
        "per_class_F1": {c: (data.get("per_class_F1", {}) or {}).get(c, 0)
                         for c in DEFECT_CLASSES},
        "per_class_Precision": {c: (data.get("per_class_Precision", {}) or {}).get(c, 0)
                                for c in DEFECT_CLASSES},
        "per_class_Recall": {c: (data.get("per_class_Recall", {}) or {}).get(c, 0)
                             for c in DEFECT_CLASSES},
    }
    result["defect_mean_iou"] = float(np.mean(list(result["per_class_IoU"].values())))
    return result


def detect_lr_restart(lr):
    """识别 warm restart: 学习率落到近谷底后又被抬回高位.

    返回 1-based 的跳变轮次; 无跳变 / 无 lr 记录时返回 None.
    背景: 自定义 WarmupCosineScheduler 的 state_dict 只存 last_epoch,
    resume 改大 epochs 后会按新总长重排余弦, LR 从谷底跳回高位.
    """
    if not lr or len(lr) < 15:
        return None
    for i in range(5, len(lr) - 6):
        if lr[i] > 0 and lr[i] < 5e-5 and lr[i] <= lr[i - 1] and lr[i] <= lr[i + 1] \
                and lr[i + 5] > 3 * lr[i] and lr[i + 5] > 1e-4:
            return i + 1
    return None


def extract_train_effect(m, cfg):
    """训练过程指标: 收敛位置 / 预算 / 稳定性 / 退化幅度.

    这是"对比训练效果"的核心 —— 同等精度下, 谁的收敛更快、更稳、预算更省.
    """
    if m is None:
        return None
    hist = m.get("history", {}) or {}
    v = list(hist.get("val_miou", []) or [])
    tl = list(hist.get("train_loss", []) or [])
    sc = list(hist.get("val_score", []) or [])
    vl = list(hist.get("val_loss", []) or [])
    lr = list(hist.get("lr", []) or [])
    if not v:
        return None

    n = min(len(v), max(len(tl), len(v)))          # 以 val_miou 为主轴
    bi = int(np.argmax(v))                          # 最佳 epoch (0-based)
    tail = v[-10:] if len(v) >= 10 else v
    cfg_epochs = (cfg or {}).get("epochs")
    patience = (cfg or {}).get("patience")

    # 早停 vs 预算耗尽: 实际 epoch 数少于配置值 => 早停触发
    if cfg_epochs and len(v) < cfg_epochs:
        stop = "早停触发"
    elif cfg_epochs:
        stop = "预算耗尽"
    else:
        stop = "未记录"

    restart_ep = detect_lr_restart(lr)
    best_val = float(v[bi])
    final_val = float(v[-1])
    return {
        "epochs": len(v),
        "cfg_epochs": cfg_epochs,
        "patience": patience,
        "min_lr": (cfg or {}).get("min_lr"),
        "seed": (cfg or {}).get("seed"),
        "resume": bool((cfg or {}).get("resume")),
        "restart_epoch": restart_ep,
        "best_epoch": bi + 1,
        "best_val": best_val,
        "final_val": final_val,
        "decay_after_best": final_val - best_val,   # <0 = 最佳后回落
        "tail_std": float(np.std(tail)) if tail else float("nan"),
        "train_loss_first": float(tl[0]) if tl else float("nan"),
        "train_loss_last": float(tl[-1]) if tl else float("nan"),
        "val_loss_min": float(np.min(vl)) if vl else float("nan"),
        "val_score_best": float(np.max(sc)) if sc else float("nan"),
        "stop_reason": stop,
        "still_improving": bool(bi >= len(v) - 3),  # 最佳 epoch 落在末尾 3 轮 = 还在变好
        "curve": {"val_miou": v, "train_loss": tl, "val_score": sc},
    }


def compute_gains(mmetrics):
    """按 GAIN_DEFS 计算所有 Δ; 缺模型则自动跳过该项."""
    out = []
    for d in GAIN_DEFS:
        a, b = mmetrics.get(d["a"]), mmetrics.get(d["b"])
        if not a or not b:
            continue
        out.append(dict(d, delta=a["mIoU"] - b["mIoU"]))
    return out


def index_carry_ratio(mmetrics):
    """光谱增益的指数承载比 = Δ_index_rgb / Δ_spectral."""
    rgb_m = mmetrics.get("rgb")
    early_m = mmetrics.get("early")
    rgb_idx = mmetrics.get("rgb_indices")
    if not (rgb_m and early_m and rgb_idx):
        return None
    denom = early_m["mIoU"] - rgb_m["mIoU"]
    if abs(denom) < 1e-6:
        return None
    return (rgb_idx["mIoU"] - rgb_m["mIoU"]) / denom


# ══════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════
def plot_overall(mmetrics, out_path, source):
    """总体指标柱状图: mIoU / OA / balance score, 带 pan 基线参考线."""
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in MODEL_ORDER if mmetrics.get(n)]
    if not names:
        return
    pan = mmetrics.get("pan")
    best = max(names, key=lambda n: mmetrics[n]["mIoU"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    specs = [("mIoU", "mIoU (test)", 3), ("OA", "OA (test)", 3),
             ("score", "Balance Score (test 重算)", 3)]
    for ax, (key, title, prec) in zip(axes, specs):
        vals = [mmetrics[n].get(key) for n in names]
        vals = [v if isinstance(v, (int, float)) else np.nan for v in vals]
        xs = np.arange(len(names))
        bars = ax.bar(xs, vals, color=[MODELS[n]["color"] for n in names],
                      edgecolor="white", linewidth=0.8)
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            ax.text(i, v + 0.012, f"{v:.{prec}f}", ha="center", va="bottom", fontsize=7.8)
        # pan 参考线 = "仅可见光" 的水平 (文字放左上角坐标区, 避免与柱值标签重叠)
        if pan and isinstance(pan.get(key), (int, float)):
            ax.axhline(pan[key], color="#7f8c8d", linestyle="--", linewidth=1.1)
            ax.text(0.02, 0.97, f"虚线 = pan 基线 {pan[key]:.3f}",
                    transform=ax.transAxes, fontsize=7.2, color="#7f8c8d",
                    va="top", ha="left")
        finite = [v for v in vals if not np.isnan(v)]
        lo, hi = min(finite), max(finite)
        pad = max(0.015, (hi - lo) * 0.35)
        ax.set_ylim(max(0.0, lo - pad * 1.6), min(1.0, hi + pad))
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs)
        labels = [f"{MODELS[n]['label']}" + (" ★" if n == best and key == "mIoU" else "")
                  for n in names]
        ax.set_xticklabels(labels, rotation=26, ha="right", fontsize=7.6)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for b, n in zip(bars, names):
            b.set_alpha(0.92)

    fig.suptitle(f"七模型总体指标对比 ({source}) — 纵轴已截断以放大差异", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class(mmetrics, out_path, source):
    """分类别 IoU 分组柱状图 (含 background)."""
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in MODEL_ORDER if mmetrics.get(n)]
    if not names:
        return
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    n = len(names)
    w = 0.82 / n

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for i, name in enumerate(names):
        vals = [mmetrics[name]["per_class_IoU_all"][c] for c in classes]
        ax.bar(x + (i - n / 2 + 0.5) * w, vals, w,
               label=MODELS[name]["label"], color=MODELS[name]["color"],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([CLS_SHORT[c] for c in classes], fontsize=10)
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"分类别 IoU ({source})", fontsize=12)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class_gain_heatmap(mmetrics, out_path, source):
    """ΔIoU 热力图 (相对 pan): 直接看出每个模型在哪些类别上有增益."""
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in MODEL_ORDER if mmetrics.get(n)]
    if not names:
        return
    pan = mmetrics.get("pan")

    rows = names
    M = np.zeros((len(rows), len(DEFECT_CLASSES)))
    for i, name in enumerate(rows):
        for j, c in enumerate(DEFECT_CLASSES):
            v = mmetrics[name]["per_class_IoU"][c]
            M[i, j] = v - pan["per_class_IoU"][c] if pan else v

    fig, ax = plt.subplots(figsize=(8.2, 1.0 + 0.58 * len(rows)))
    if pan:
        vmax = max(0.05, float(np.max(np.abs(M))) * 1.05)
        im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        cl = "ΔIoU (相对 pan 基线)"
        zero_row = [i for i, n in enumerate(rows) if n == "pan"]
        if zero_row:
            for j in range(len(DEFECT_CLASSES)):
                ax.text(j, zero_row[0], "基线", ha="center", va="center",
                        fontsize=8, color="#34495e")
    else:
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        cl = "IoU (绝对值)"

    for i in range(len(rows)):
        for j in range(len(DEFECT_CLASSES)):
            if pan and MODELS[rows[i]]["short"] == "pan":
                continue
            txt = f"{M[i, j]:+.3f}" if pan else f"{M[i, j]:.3f}"
            dark = abs(M[i, j]) > (0.6 * float(np.max(np.abs(M))) if pan else 0.6)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if dark else "black")

    ax.set_xticks(range(len(DEFECT_CLASSES)))
    ax.set_xticklabels([CLS_SHORT[c] for c in DEFECT_CLASSES], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MODELS[n]["label"] for n in rows], fontsize=9)
    ax.set_title(f"分类别增益热力图 ({source})", fontsize=12, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label(cl, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_gain_decomposition(gains, carry, out_path, source):
    """增益分解水平条形图 (融合方式 + 光谱指数)."""
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not gains:
        print("  [SKIP] 增益分解: 可用模型不足以构成对照")
        return

    labels, deltas, colors, groups = [], [], [], []
    for g in gains:
        labels.append(f"{g['short']}: {MODELS[g['a']]['short']} − {MODELS[g['b']]['short']}")
        deltas.append(g["delta"])
        if g["delta"] > SIG_DELTA:
            colors.append("#1e8449")
        elif g["delta"] < -SIG_DELTA:
            colors.append("#c0392b")
        else:
            colors.append("#95a5a6")
        groups.append(g["group"])

    fig, ax = plt.subplots(figsize=(10.5, 1.2 + 0.52 * len(gains)))
    ys = np.arange(len(gains))[::-1]
    ax.barh(ys, deltas, color=colors, height=0.55, edgecolor="white", linewidth=0.4)
    xmax = max(0.01, float(np.max(np.abs(deltas))) * 1.25)
    ax.set_xlim(-xmax, xmax)
    for y, d in zip(ys, deltas):
        ax.text(d + (xmax * 0.02 if d >= 0 else -xmax * 0.02), y, f"{d:+.4f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    # 显著性带: |Δ| < 2 点
    ax.axvspan(-SIG_DELTA, SIG_DELTA, color="#f2f3f4", zorder=0)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.6)
    # 分组分隔线
    prev = None
    for y, g in zip(ys, groups):
        if prev is not None and g != prev:
            ax.axhline(y + 0.5, color="#bdc3c7", linewidth=0.8, linestyle=":")
        prev = g
    note = (f"灰底带 = |Δ| < {SIG_DELTA:.2f} (无显著差异)   "
            f"指数承载比 = {carry * 100:.1f}%" if carry is not None
            else f"灰底带 = |Δ| < {SIG_DELTA:.2f} (无显著差异)")
    ax.set_xlabel(f"Δ mIoU ({source})")
    ax.set_title(f"增益分解 ({source})\n{note}", fontsize=11)
    ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_training_curves(train_effects, out_path):
    """训练曲线对比: val mIoU / train loss / val score, 标注各自最佳 epoch."""
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    av = {k: v for k, v in train_effects.items() if v}
    if not av:
        print("  [SKIP] 训练曲线: 无 history")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
    panels = [("val_miou", "验证集 mIoU (收敛曲线)"), ("train_loss", "训练损失"),
              ("val_score", "验证集 Balance Score")]
    for ax, (key, title) in zip(axes, panels):
        for i, name in enumerate(MODEL_ORDER):
            te = av.get(name)
            if not te:
                continue
            arr = te["curve"][key]
            if not arr:
                continue
            color = MODELS[name]["color"]
            xs = np.arange(1, len(arr) + 1)
            ax.plot(xs, arr, color=color, linewidth=1.3, alpha=0.92,
                    label=MODELS[name]["label"])
            if key == "val_miou":
                be = te["best_epoch"]
                ax.scatter([be], [arr[be - 1]], s=42, color=color,
                           edgecolor="white", linewidth=0.8, zorder=5)
                # 上下交错标注 + 半透明白底, 避免最佳 epoch 相近或曲线密集处压字
                dy = 9 if i % 2 == 0 else -12
                ax.annotate(f"ep{be}", (be, arr[be - 1]), fontsize=6.4, color=color,
                            textcoords="offset points", xytext=(4, dy), zorder=6,
                            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                      alpha=0.78, edgecolor="none"))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(fontsize=7.2, ncol=2, loc="lower right")
    fig.suptitle("训练效果对比: 曲线形状 = 收敛速度与稳定性, 圆点 = 最佳 epoch (val mIoU)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_convergence_efficiency(train_effects, mmetrics, out_path, source):
    """左: 训练预算哑铃图 (best epoch → 总 epoch); 右: best val vs test."""
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in MODEL_ORDER if train_effects.get(n)]
    if not names:
        print("  [SKIP] 收敛效率图: 无可用 history")
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.5, 5.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # ── 左: 哑铃图 ──
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names):
        te = train_effects[name]
        c = MODELS[name]["color"]
        total = te["epochs"]
        be = te["best_epoch"]
        a1.plot([be, total], [y, y], color=c, linewidth=1.6, alpha=0.55, zorder=2)
        a1.scatter([be], [y], s=68, color=c, zorder=4, edgecolor="white", linewidth=0.9)
        a1.scatter([total], [y], s=46, facecolor="white", edgecolor=c,
                   linewidth=1.4, zorder=4)
        tag = " (早停)" if te["stop_reason"] == "早停触发" else " (预算耗尽)"
        a1.text(max(total, be) + 3, y, f"best ep{be}/{total}{tag}", va="center",
                fontsize=8, color="#2c3e50")
    a1.set_yticks(ys)
    a1.set_yticklabels([MODELS[n]["label"] for n in names], fontsize=9)
    a1.set_xlabel("epoch")
    a1.set_xlim(0, max(train_effects[n]["epochs"] for n in names) * 1.32)
    a1.set_title("训练预算利用: ● 最佳 epoch → ○ 实际结束 epoch", fontsize=11)
    a1.grid(axis="x", alpha=0.25, linewidth=0.6)

    # ── 右: best val vs test ──
    x = np.arange(len(names))
    best_vals = [train_effects[n]["best_val"] for n in names]
    tests = [mmetrics[n]["mIoU"] if mmetrics.get(n) else np.nan for n in names]
    w = 0.38
    a2.bar(x - w / 2, best_vals, w, color="#95a5a6",
           edgecolor="white", linewidth=0.5)
    a2.bar(x + w / 2, tests, w,
           color=[MODELS[n]["color"] for n in names], edgecolor="white", linewidth=0.5)
    for i, (bv, tv) in enumerate(zip(best_vals, tests)):
        a2.text(i - w / 2, bv + 0.012, f"{bv:.3f}", ha="center", fontsize=7.2)
        if not np.isnan(tv):
            a2.text(i + w / 2, tv + 0.012, f"{tv:.3f}", ha="center", fontsize=7.2)
    lo = min([v for v in best_vals + list(tests) if not np.isnan(v)])
    hi = max([v for v in best_vals + list(tests) if not np.isnan(v)])
    a2.set_ylim(max(0, lo - 0.06), hi + 0.05)
    a2.set_xticks(x)
    a2.set_xticklabels([MODELS[n]["short"] for n in names], rotation=22, ha="right",
                       fontsize=9)
    a2.set_title(f"选优分 vs 测试指标 (灰 = best val, 彩色 = {source})", fontsize=11)
    a2.grid(axis="y", alpha=0.25, linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
# HTML 报告
# ══════════════════════════════════════════════════════════════
HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
         margin: 0; padding: 24px; background: #f4f6f7; color: #2c3e50; }
  .container { max-width: 1380px; margin: 0 auto; background: #fff; padding: 32px 36px;
               border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
  h1 { font-size: 26px; border-bottom: 3px solid #377eb8; padding-bottom: 12px; }
  h2 { font-size: 19px; color: #34495e; margin-top: 34px;
       border-left: 4px solid #377eb8; padding-left: 10px; }
  h3 { font-size: 15px; color: #34495e; margin: 18px 0 8px; }
  p.sub { color: #7f8c8d; font-size: 13px; margin-top: -6px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0 6px; font-size: 13px; }
  th, td { border: 1px solid #dfe4e6; padding: 8px 10px; text-align: center; }
  th { background: #34495e; color: #fff; font-weight: 600; }
  td.rowname, th.rowname { text-align: left; background: #f8f9fa; font-weight: 600; }
  tr:nth-child(even) td { background: #fbfcfc; }
  .imgbox { text-align: center; margin: 16px 0 4px; }
  .imgbox img { max-width: 100%; border: 1px solid #e1e5e8; border-radius: 6px; }
  .caption { color: #7f8c8d; font-size: 12px; text-align: center; margin-bottom: 18px; }
  .keybox { background: #eef5fb; border-left: 5px solid #377eb8; padding: 16px 20px;
            border-radius: 6px; margin: 20px 0; }
  .keybox h3 { margin: 0 0 8px; color: #21618c; }
  .warn { background: #fef5e7; border-left: 5px solid #f39c12; padding: 14px 18px;
          border-radius: 6px; margin: 16px 0; font-size: 13.5px; }
  .ok { background: #eafaf1; border-left: 5px solid #1e8449; padding: 14px 18px;
        border-radius: 6px; margin: 16px 0; font-size: 13.5px; }
  ul { line-height: 1.85; font-size: 13.5px; }
  li { margin-bottom: 4px; }
  code { background: #f4f6f7; padding: 1px 5px; border-radius: 3px;
         font-family: Consolas, monospace; font-size: 12.5px; }
  .footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid #ecf0f1;
            color: #95a5a6; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
"""


def _table(header_cells, rows_html, first_col="模型"):
    head = f'<tr><th class="rowname">{first_col}</th>{header_cells}</tr>'
    return f"<table>{head}{rows_html}</table>"


def generate_html_report(ctx):
    """组装 HTML 报告. 所有动态内容先在函数内算成字符串, 避免 f-string 内出现字面量花括号."""
    mm = ctx["models"]
    te = ctx["train_effects"]
    params = ctx["params"]
    gains = ctx["gains"]
    carry = ctx["carry"]
    source = ctx["source"]
    root = ctx["root"]
    out_dir = ctx["out_dir"]

    names = [n for n in MODEL_ORDER if mm.get(n)]
    best = max(names, key=lambda n: mm[n]["mIoU"]) if names else None
    pan = mm.get("pan")

    seeds = [te[n]["seed"] for n in names if te.get(n) and te[n].get("seed") is not None]
    if seeds and len(set(seeds)) == 1:
        seed_txt = f"seed={seeds[0]}"
    elif seeds:
        seed_txt = "seed=" + "/".join(str(s) for s in dict.fromkeys(seeds))
    else:
        seed_txt = "seed 未记录"

    # ── 0. 一句话结论 ──
    if best:
        lead = (f"<strong>{MODELS[best]['label']}</strong> 以 test mIoU "
                f"<strong>{mm[best]['mIoU']:.4f}</strong> 居首")
        if pan and best != "pan":
            d = mm[best]["mIoU"] - pan["mIoU"]
            lead += (f"，相对 pan 基线 {dspan(d)}（{rel_pct(d, pan['mIoU'])}）")
        lead += f"；参数量 {params[best][0] / 1e6:.2f}M，训练 {te[best]['epochs']} epoch" \
                f"（最佳 ep{te[best]['best_epoch']}）。" if te.get(best) else "。"
    else:
        lead = "没有可用模型。"

    # ── 1. 实验矩阵 ──
    matrix_rows = ""
    for name in names:
        cfg = ctx["cfg"].get(name) or {}
        p_n, p_src = params[name]
        t = te.get(name) or {}
        matrix_rows += (
            "<tr>"
            f"<td class='rowname'>{MODELS[name]['label']}</td>"
            f"<td>{MODELS[name]['family']}</td>"
            f"<td style='text-align:left'>{MODELS[name]['input']}</td>"
            f"<td>{p_n / 1e6:.2f}M<br><span style='color:#95a5a6;font-size:11px'>{p_src}</span></td>"
            f"<td>{t.get('cfg_epochs') or 'N/A'}<br>"
            f"<span style='color:#95a5a6;font-size:11px'>patience {cfg.get('patience') or 'N/A'}"
            f" / min_lr {cfg.get('min_lr') or 'N/A'}"
            f" / seed {t.get('seed') or 'N/A'}</span></td>"
            f"<td><code>{MODELS[name]['dir']}</code></td>"
            "</tr>")
    matrix_head = ("<th>模型</th><th>实验分组</th><th>输入</th><th>参数量</th>"
                   "<th>训练预算</th><th>结果目录</th>")
    matrix_tbl = f"<table><tr>{matrix_head}</tr>{matrix_rows}</table>"

    # ── 2. 总体指标 ──
    head_cells = "".join(
        f'<th style="color:{MODELS[n]["color"]};background:#34495e">{MODELS[n]["label"]}</th>'
        for n in names)
    overall_keys = [("score", "Balance Score (test 重算)", 4), ("mIoU", "mIoU", 4),
                    ("OA", "OA", 4), ("defect_mean_iou", "缺陷类平均 IoU", 4)]
    abs_rows = ""
    for key, label, prec in overall_keys:
        cells = ""
        for n in names:
            v = mm[n].get(key)
            txt = fmt(v, prec)
            if key == "mIoU" and n == best:
                txt = f"<strong>{txt}</strong> ★"
            cells += f"<td>{txt}</td>"
        abs_rows += f"<tr><td class='rowname'>{label}</td>{cells}</tr>"
    abs_tbl = _table(head_cells, abs_rows, first_col="指标")

    # Δ vs pan
    others = [n for n in names if n != "pan"]
    if pan and others:
        dh = "".join(f'<th>{MODELS[n]["label"]}</th>' for n in others)
        drows = ""
        for key, label, _ in overall_keys:
            cells = ""
            for n in others:
                a, b = mm[n].get(key), pan.get(key)
                cells += f"<td>{dspan(a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else 'N/A'}</td>"
            drows += f"<tr><td class='rowname'>{label} − pan</td>{cells}</tr>"
        delta_tbl = _table(dh, drows, first_col="Δ (相对 pan 基线)")
    else:
        delta_tbl = "<p class='caption'>缺少 pan 基线，无法计算 Δ。</p>"

    # ── 3. 分类别 ──
    c_rows = ""
    for c in CLASS_NAMES:
        cells = "".join(f"<td>{fmt(mm[n]['per_class_IoU_all'][c])}</td>" for n in names)
        c_rows += f"<tr><td class='rowname'>{c}</td>{cells}</tr>"
    class_tbl = _table(head_cells, c_rows, first_col="类别")

    cd_rows = ""
    if pan:
        for c in DEFECT_CLASSES:
            cells = "".join(
                f"<td>{dspan(mm[n]['per_class_IoU'][c] - pan['per_class_IoU'][c])}</td>"
                for n in others)
            cd_rows += f"<tr><td class='rowname'>{c}</td>{cells}</tr>"
        dh2 = "".join(f'<th>{MODELS[n]["label"]}</th>' for n in others)
        class_delta_tbl = _table(dh2, cd_rows, first_col="Δ IoU (相对 pan)")
    else:
        class_delta_tbl = ""

    # ── 4. 训练效果 ──
    t_head = ("<tr><th class='rowname'>模型</th><th>预算 / 实际 epoch</th>"
              "<th>最佳 epoch</th><th>结束方式</th><th>best val mIoU</th>"
              "<th>final val mIoU</th><th>最佳后回落</th><th>尾段 σ (末 10 轮)</th>"
              "<th>train loss 首 → 末</th><th>test mIoU</th><th>test − best val</th></tr>")
    t_rows = ""
    for n in names:
        t = te.get(n)
        if not t:
            t_rows += (f"<tr><td class='rowname'>{MODELS[n]['label']}</td>"
                       + "<td>N/A</td>" * 10 + "</tr>")
            continue
        tm = mm.get(n)
        gap = (tm["mIoU"] - t["best_val"]) if tm else None
        tail_ok = "✔" if t["tail_std"] < 0.01 else "⚠"
        stop_tag = t["stop_reason"]
        if t["still_improving"]:
            stop_tag += "<br><span style='color:#c0392b;font-size:11px'>最佳在末 3 轮, 仍在变好</span>"
        flags = []
        if t.get("resume"):
            flags.append("断点续训")
        if t.get("restart_epoch"):
            flags.append(f"LR 重启@ep{t['restart_epoch']}")
        if flags:
            stop_tag += "<br><span style='color:#8e44ad;font-size:11px'>" \
                        + " / ".join(flags) + "</span>"
        t_rows += (
            "<tr>"
            f"<td class='rowname'>{MODELS[n]['label']}</td>"
            f"<td>{t['cfg_epochs'] if t['cfg_epochs'] else 'N/A'} / {t['epochs']}</td>"
            f"<td>{t['best_epoch']}</td>"
            f"<td>{stop_tag}</td>"
            f"<td>{t['best_val']:.4f}</td>"
            f"<td>{t['final_val']:.4f}</td>"
            f"<td>{dspan(t['decay_after_best'])}</td>"
            f"<td>{t['tail_std']:.4f} {tail_ok}</td>"
            f"<td>{fmt(t['train_loss_first'], 3)} → {fmt(t['train_loss_last'], 3)}</td>"
            f"<td>{fmt(tm['mIoU']) if tm else 'N/A'}</td>"
            f"<td>{dspan(gap) if gap is not None else 'N/A'}</td>"
            "</tr>")
    train_tbl = f"<table>{t_head}{t_rows}</table>"

    # 训练预算一致性检测
    budgets = sorted({te[n]["cfg_epochs"] for n in names if te.get(n) and te[n]["cfg_epochs"]})
    mismatched = len(budgets) > 1
    if mismatched:
        budget_note = (
            '<div class="warn"><strong>训练预算不一致（自动检测）：</strong> 各模型的配置 epoch 为 '
            + " / ".join(str(b) for b in budgets) +
            "。跨预算的两两 Δ 混合了「输入/结构差异」与「训练时长差异」两种效应，"
            "不能整体归因于前者。若某模型在预算耗尽时最佳 epoch 仍落在末 3 轮（见上表末列标记），"
            "说明它仍在变好，差异主要来自训练不足。要对齐口径，应把训练预算统一后重训再比较。</div>")
    else:
        budget_note = ('<div class="ok"><strong>训练预算一致：</strong> 所有模型配置 epoch 相同，'
                       "跨模型 Δ 可较干净地归因于输入/结构差异。</div>")

    # 训练稳定性最好的模型
    stable = None
    cand = [n for n in names if te.get(n)]
    if cand:
        stable = min(cand, key=lambda n: te[n]["tail_std"])

    # ── 5. 增益分解 ──
    g_rows = ""
    for g in gains:
        g_rows += (
            "<tr>"
            f"<td class='rowname'>{g['key']}</td>"
            f"<td>{MODELS[g['a']]['label']} − {MODELS[g['b']]['label']}</td>"
            f"<td>{dspan(g['delta'])}</td>"
            f"<td>{verdict(g['delta'])}</td>"
            f"<td style='text-align:left'>{g['desc']}</td>"
            "</tr>")
    if carry is not None:
        g_rows += (
            "<tr><td class='rowname'>指数承载比</td>"
            "<td>(rgb_idx − RGB) / (early − RGB)</td>"
            f"<td>{carry * 100:.1f}%</td><td>—</td>"
            "<td style='text-align:left'>光谱增益中可被 4 个 NDI 显式承载的比例</td></tr>")
    gain_tbl = (f"<table><tr><th class='rowname'>增益项</th><th>对比</th><th>Δ mIoU</th>"
                f"<th>判定</th><th>含义</th></tr>{g_rows}</table>")

    # ── 6. 自动结论 ──
    concl = []
    ranked = sorted(names, key=lambda n: -mm[n]["mIoU"])
    concl.append("<li><strong>总体排序（test mIoU）：</strong>" + " &gt; ".join(
        f"{MODELS[n]['label']} {mm[n]['mIoU']:.4f}" for n in ranked) + "</li>")

    if pan and best and best != "pan":
        d = mm[best]["mIoU"] - pan["mIoU"]
        concl.append(
            f"<li><strong>最优模型：</strong>{MODELS[best]['label']}，mIoU {mm[best]['mIoU']:.4f}，"
            f"相对 pan 基线 {dspan(d)}（{rel_pct(d, pan['mIoU'])}），"
            f"判定 <strong>{verdict(d)}</strong>。参数量 {params[best][0] / 1e6:.2f}M。</li>")

    # 融合方式
    e_m, i_m, n_m = mm.get("early"), mm.get("intermediate"), mm.get("intermediate_narrow")
    if e_m and i_m:
        d = i_m["mIoU"] - e_m["mIoU"]
        concl.append(f"<li><strong>融合方式（宽塔 vs 早期融合）：</strong>"
                     f"中期融合 {i_m['mIoU']:.4f} vs 早期融合 {e_m['mIoU']:.4f}，"
                     f"Δ {dspan(d)}，判定 <strong>{verdict(d)}</strong>；"
                     f"参数量 {params['intermediate'][0] / 1e6:.2f}M vs "
                     f"{params['early'][0] / 1e6:.2f}M。"
                     + marginal_note(d) + "</li>")
    if n_m and i_m:
        d = n_m["mIoU"] - i_m["mIoU"]
        concl.append(f"<li><strong>容量消融（窄塔 0.5 vs 宽塔 1.0）：</strong>"
                     f"{n_m['mIoU']:.4f} vs {i_m['mIoU']:.4f}，Δ {dspan(d)}，"
                     f"判定 <strong>{verdict(d)}</strong>；参数 {params['intermediate_narrow'][0] / 1e6:.2f}M vs "
                     f"{params['intermediate'][0] / 1e6:.2f}M。"
                     + ("参数减 33% 而精度更高，说明数据规模下双塔容量存在冗余。"
                        if d > SIG_DELTA else "") + "</li>")
    if n_m and e_m:
        d = n_m["mIoU"] - e_m["mIoU"]
        tail = ("该差异小于 2 点，不宜宣称融合方式本身带来增益。"
                if abs(d) < SIG_DELTA else marginal_note(d))
        concl.append(f"<li><strong>融合方式（窄塔 vs 早期融合）：</strong>Δ {dspan(d)}，"
                     f"判定 <strong>{verdict(d)}</strong>；参数 "
                     f"{params['intermediate_narrow'][0] / 1e6:.2f}M vs {params['early'][0] / 1e6:.2f}M。"
                     + tail + "</li>")

    # 指数
    e2, e3 = mm.get("ms_indices"), mm.get("rgb_indices")

    def budget_epoch_note(a_key, b_key, a_label, b_label):
        """两模型 epoch 数不同时给中性口径提示, 不预设谁占优."""
        ta, tb = te.get(a_key), te.get(b_key)
        if not ta or not tb or ta["epochs"] == tb["epochs"]:
            return ""
        if ta["epochs"] < tb["epochs"]:
            note = f" 注意 {a_label} 训练 {ta['epochs']} epoch，少于 {b_label} {tb['epochs']} epoch，预算不对等，该差值偏保守。"
        else:
            note = f" 注意 {a_label} 训练 {ta['epochs']} epoch，多于 {b_label} {tb['epochs']} epoch，预算不对等，其优势可能部分来自更长训练。"
        if ta.get("resume"):
            note += "（该模型为断点续训）"
        return note

    if e2 and e_m:
        d = e2["mIoU"] - e_m["mIoU"]
        extra = budget_epoch_note("ms_indices", "early", "E2", "早期融合")
        concl.append(f"<li><strong>E2 指数增益：</strong>ms+idx {e2['mIoU']:.4f} vs early "
                     f"{e_m['mIoU']:.4f}，Δ {dspan(d)}，判定 <strong>{verdict(d)}</strong>。"
                     f"即「在原始 6 通道之上再显式给 4 个 NDI」，本次{gain_phrase(d)}。{extra}</li>")
    if e3 and mm.get("rgb"):
        d = e3["mIoU"] - mm["rgb"]["mIoU"]
        extra = budget_epoch_note("rgb_indices", "rgb", "E3", "RGB 基线")
        if abs(d) < SIG_DELTA:
            e3_phrase = "在当前证据下未能成立（差异未达 2 点显著阈值）"
        elif d > 0:
            e3_phrase = "方向得到支持（可见光+4 个 NDI 已逼近全多光谱）"
        else:
            e3_phrase = "在当前证据下不成立"
        concl.append(f"<li><strong>E3 可见光+指数：</strong>rgb+idx {e3['mIoU']:.4f} vs RGB "
                     f"{mm['rgb']['mIoU']:.4f}，Δ {dspan(d)}，判定 <strong>{verdict(d)}</strong>。"
                     f"「传感器可降配」假设{e3_phrase}。{extra}</li>")
    if carry is not None:
        concl.append(f"<li><strong>指数承载比：</strong>{carry * 100:.1f}% —— "
                     f"光谱增益中可被 4 个 NDI 承载的比例。"
                     + ("为负且绝对值远大于 100%，说明指数通道并未「承载」光谱增益，"
                        "反而引入了额外损失；E2/E3 各自的方向见上两条。"
                        "另外分母 (early − RGB) 本身只有 0.01 量级，"
                        "该比值的解释力有限，只宜作方向性参考。"
                        if carry < 0 else "") + "</li>")

    # 收敛与稳定性
    if stable:
        concl.append(f"<li><strong>训练稳定性：</strong>{MODELS[stable]['label']} 尾段波动最小"
                     f"（σ={te[stable]['tail_std']:.4f}）；"
                     + "、".join(f"{MODELS[n]['label']} σ={te[n]['tail_std']:.4f}"
                                for n in cand if n != stable) + "。尾段 σ &lt; 0.01 视为收敛平稳。</li>")
    exhausted = [n for n in cand if te[n]["stop_reason"] == "预算耗尽"]
    early_stopped = [n for n in cand if te[n]["stop_reason"] == "早停触发"]
    if cand:
        txt = []
        if exhausted:
            txt.append("预算耗尽：" + "、".join(f"{MODELS[n]['short']}({te[n]['epochs']}ep)" for n in exhausted))
        if early_stopped:
            txt.append("早停触发：" + "、".join(
                f"{MODELS[n]['short']}(best ep{te[n]['best_epoch']}, 共 {te[n]['epochs']}ep)"
                for n in early_stopped))
        concl.append("<li><strong>训练预算利用：</strong>" + "；".join(txt) + "。"
                     "最佳 epoch 越靠近末尾，说明该模型在预算耗尽时仍有提升空间；"
                     "跨预算的 mIoU 不宜直接对齐比较（见第 7 节）。</li>")

    # 类别级
    if pan:
        for c in DEFECT_CLASSES:
            vals = [(n, mm[n]["per_class_IoU"][c]) for n in names]
            top = max(vals, key=lambda kv: kv[1])
            best_gain = max(vals, key=lambda kv: kv[1] - pan["per_class_IoU"][c])
            d = best_gain[1] - pan["per_class_IoU"][c]
            concl.append(f"<li><strong>{CLS_SHORT[c]}：</strong>最高 {MODELS[top[0]]['label']} "
                         f"{top[1]:.4f}；相对 pan 增益最大的是 {MODELS[best_gain[0]]['label']} "
                         f"（{dspan(d)}）。</li>")

    concl_html = "<ul>" + "".join(concl) + "</ul>"

    # ── 7. 公平性与局限 ──
    caveats = [
        "<li><strong>相同项：</strong>数据划分（train 214 / val 60 / test 34，seed=42 场景级分层）、损失"
        "（Focal v4 alpha + Lovasz + Boundary）、batch=4、crop=512、AdamW lr=1e-3 + Warmup/Cosine、"
        "EMA 0.998、梯度过剪 1.0、增强策略、滑窗验证与 balance score 选优。</li>",
        "<li><strong>不同项（须如实报告）：</strong>训练预算（配置 epoch）与 patience、min_lr 各模型不同；"
        "中期融合双塔结构参数量显著更大。见第 1、4 节。</li>",
        f"<li><strong>单 seed：</strong>所有结果为 {seed_txt} 单次运行，test 仅 34 个场景。"
        "两两差异 &lt; 2 个 mIoU 点时一律按「无显著差异」表述；如需支撑论文级结论，"
        "应补 3 seed 重复 + 34 场景逐场景配对检验（Wilcoxon signed-rank）。</li>",
        "<li><strong>判定阈值：</strong>本报告以 |Δ| &lt; 0.02（2 个 mIoU 点）为「无显著差异」，"
        "在图表中以灰色呈现。</li>",
        "<li><strong>test 集口径：</strong>test 指标在训练完成后一次性评估（metrics.json 的 test 段），"
        "未参与选优，无信息泄露。</li>",
    ]
    if mismatched:
        caveats.insert(1, "<li><strong>★ 训练预算不对等是本报告最大口径风险：</strong>"
                          f"配置 epoch 分别为 {' / '.join(str(b) for b in budgets)}。"
                          "跨预算的两两 Δ 混合了「输入/结构差异」与「训练时长差异」两种效应："
                          "训练更久的模型可能因额外轮次受益，训练更短的模型可能尚未收敛，"
                          "不能把 Δ 整体归因于前者。下结论前建议统一预算、从零重训后再比较。</li>")
    resume_names = [n for n in names if te.get(n) and te[n].get("resume")]
    if resume_names:
        resume_items = []
        for n in resume_names:
            r = te[n].get("restart_epoch")
            resume_items.append(MODELS[n]["short"] + (f"（LR 重启@ep{r}）" if r else ""))
        caveats.insert(1, "<li><strong>★ 部分模型为断点续训（warm restart）：</strong>"
                      + "、".join(resume_items) + "。这些模型从既有 checkpoint 继续训练，"
                      "学习率调度按本次命令的总轮数重排，曲线含一次计划外的 LR 抬升，"
                      "不能与「从零同预算训练」的曲线直接比较；其精度优劣需在统一从零重训后复核。</li>")
    caveats_html = "<ul>" + "".join(caveats) + "</ul>"

    # ── 拼接 ──
    body = []
    body.append(f"<h1>七模型对比与训练效果评估报告</h1>")
    body.append(f"<p class='sub'>结果来源：<code>{root}</code> ｜ 指标来源：<strong>{source}</strong> ｜ "
                f"{seed_txt} ｜ 生成时间：{ctx['timestamp']} ｜ 脚本：compare_fusion.py (v5.5)</p>")
    body.append(f"<div class='keybox'><h3>一句话结论</h3><p style='margin:0;font-size:14px'>{lead}</p></div>")

    if source != "test":
        body.append("<div class='warn'><strong>口径提醒：</strong>本次使用 <strong>val</strong> 指标，"
                    "取各模型<b>最后一轮</b>的验证结果（final_val），并非最佳 checkpoint。"
                    "test 口径是 best_model.pth 在独立 test 集上一次性评估，两者不可直接对比；"
                    "要写结论请用 <code>--source test</code>。</div>")

    body.append("<h2>1. 实验矩阵与模型概览</h2>")
    body.append(matrix_tbl)
    body.append("<p class='caption'>匹配实验结果后按 <code>--source test</code> 生成的最终报告；"
                "参数量用于判断精度差异是否可能只来自容量。</p>")

    body.append(f"<h2>2. 总体指标（{source}）</h2>")
    body.append(abs_tbl)
    body.append(delta_tbl)
    body.append("<p class='caption'>Balance Score = 0.40·mIoU + 0.30·meanDefectF1 + 0.15·meanDefectP "
                "+ 0.15·meanDefectR − 0.10·meanDefect|P−R|，在 test 的 P/R 上重算；"
                "括号外的 Δ 相对 pan 基线，灰色表示 |Δ| &lt; 2 个 mIoU 点。</p>")
    body.append("<div class='imgbox'><img src='overall_comparison.png' alt='总体指标'></div>")
    body.append("<p class='caption'>mIoU / OA / Balance Score；虚线为 pan 基线水平，纵轴已截断以放大差异。</p>")

    body.append(f"<h2>3. 分类别 IoU（{source}）</h2>")
    body.append(class_tbl)
    if class_delta_tbl:
        body.append("<h3>相对 pan 基线的分类别增益</h3>")
        body.append(class_delta_tbl)
        body.append("<p class='caption'>这是「光谱增益按病害类型分化」结论的直接证据："
                    "同一模型在不同类别上的 Δ 可能一正一负。</p>")
    body.append("<div class='imgbox'><img src='per_class_comparison.png' alt='分类别 IoU'></div>")
    body.append("<p class='caption'>四类（含背景）IoU 分组对比。</p>")
    body.append("<div class='imgbox'><img src='per_class_gain_heatmap.png' alt='分类别增益热力图'></div>")
    body.append("<p class='caption'>绿=优于 pan，红=劣于 pan，颜色范围以绝对值最大者对称定标。</p>")

    body.append("<h2>4. 训练效果（收敛过程、稳定性、训练预算）</h2>")
    body.append(train_tbl)
    body.append(budget_note)
    body.append("<div class='imgbox'><img src='training_curves_comparison.png' alt='训练曲线对比'></div>")
    body.append("<p class='caption'>左：val mIoU 收敛曲线（圆点=各模型最佳 epoch）；中：train loss；"
                "右：val Balance Score。曲线越早抬升、越平，说明收敛越快越稳。</p>")
    body.append("<div class='imgbox'><img src='convergence_efficiency.png' alt='收敛效率'></div>")
    body.append("<p class='caption'>左：训练预算哑铃图（● 最佳 epoch → ○ 实际结束），"
                "越靠左说明用更少 epoch 达到最佳；右：best val mIoU 与 test mIoU 的对照，"
                "两者差距大说明验证-测试分布有偏移。</p>")

    body.append(f"<h2>5. 增益分解（{source}）</h2>")
    body.append(gain_tbl)
    body.append("<div class='imgbox'><img src='gain_decomposition.png' alt='增益分解'></div>")
    body.append("<p class='caption'>绿色=正向增益，红色=负向，灰色=|Δ| &lt; 2 点（无显著差异）；"
                "灰底竖带即无显著差异区间。</p>")

    body.append("<h2>6. 结论（按数据自动生成）</h2>")
    body.append(concl_html)

    body.append("<h2>7. 公平性声明与局限</h2>")
    body.append(caveats_html)

    body.append("<div class='footer'>Generated by compare_fusion.py (v5.5, 七模型 + 训练效果) ｜ "
                "MS-CrackSeg Project ｜ 数字同时存于 comparison_summary.json</div>")
    body.append("</div></body></html>")

    html = HTML_HEAD.replace("__TITLE__", "七模型对比与训练效果评估报告") + "\n".join(body)
    out_path = out_dir / "fusion_comparison_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="七模型对比 + 训练效果评估 (v5.5)")
    parser.add_argument("--source", choices=["val", "test"], default="test",
                        help="指标来源 (默认 test)")
    parser.add_argument("--outputs-root", default=None,
                        help="结果根目录 (缺省优先 <ROOT>/outputV5, 否则 <ROOT>/outputs)")
    parser.add_argument("--out-dir", default=None,
                        help="报告输出目录 (缺省 <outputs-root>/comparison/seven_models)")
    for key, flag in [("pan", "--pan"), ("rgb", "--rgb"), ("early", "--early"),
                      ("intermediate", "--intermediate"),
                      ("intermediate_narrow", "--intermediate-narrow"),
                      ("ms_indices", "--ms-indices"),
                      ("rgb_indices", "--rgb-indices")]:
        parser.add_argument(flag, default=None, dest=key,
                            help=f"{MODELS[key]['label']} 的 metrics.json 路径")
    args = parser.parse_args()

    root, root_src = resolve_outputs_root(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "comparison" / "seven_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("七模型对比 + 训练效果评估 (v5.5)")
    print(f"  结果根目录: {root}   ({root_src})")
    print(f"  输出目录:   {out_dir}")
    print(f"  指标来源:   {args.source}"
          + ("   (val 取最后一轮 final_val, 非最佳 checkpoint)" if args.source != "test" else ""))
    print("=" * 78)

    paths = {}
    full = {}
    models_metrics = {}
    cfg = {}
    te = {}
    params = {}

    for name in MODEL_ORDER:
        p = Path(getattr(args, name)) if getattr(args, name) else \
            root / MODELS[name]["dir"] / "metrics.json"
        paths[name] = p
        m = load_json(p)
        full[name] = m
        mmx = extract_key_metrics(m, args.source)
        if mmx is None and args.source == "test":
            mmx = extract_key_metrics(m, "val")
        models_metrics[name] = mmx
        cfg[name] = load_json(p.parent / "config_used.json") or {}
        te[name] = extract_train_effect(m, cfg[name])
        params[name] = load_n_params(name, p)
        status = "OK" if mmx else "缺失"
        print(f"  [{MODELS[name]['short']:10s}] {p}  ->  {status}")

    available = [n for n in MODEL_ORDER if models_metrics[n]]
    if not available:
        print("\n!! 没有任何模型的 metrics.json 可读. 先用 --outputs-root 指向正确的结果目录, "
              "或先跑 train*.py.")
        return 0

    # ── 控制台摘要 ──
    print(f"\n可用模型 {len(available)}/7: "
          f"{[MODELS[n]['short'] for n in available]}")
    print(f"\n{'模型':<22s}{'mIoU':>8s}{'OA':>8s}{'score':>8s}"
          f"{'crack':>8s}{'bio':>8s}{'flk':>8s}  {'epochs':>7s}{'best':>6s}")
    print("-" * 92)
    for n in available:
        mm = models_metrics[n]
        t = te.get(n) or {}
        print(f"{MODELS[n]['label']:<22s}"
              f"{mm['mIoU']:>8.4f}{mm['OA']:>8.4f}"
              f"{(mm['score'] if isinstance(mm['score'], (int, float)) else float('nan')):>8.4f}"
              f"{mm['per_class_IoU']['crack']:>8.4f}"
              f"{mm['per_class_IoU']['biological_attachment']:>8.4f}"
              f"{mm['per_class_IoU']['flaking']:>8.4f}"
              f"{t.get('epochs', 0):>9d}{t.get('best_epoch', 0):>6d}")

    gains = compute_gains(models_metrics)
    carry = index_carry_ratio(models_metrics)

    # ── 图表 ──
    print("\n--- 生成图表 ---")
    try:
        plot_overall(models_metrics, out_dir / "overall_comparison.png", args.source)
        plot_per_class(models_metrics, out_dir / "per_class_comparison.png", args.source)
        plot_per_class_gain_heatmap(models_metrics, out_dir / "per_class_gain_heatmap.png",
                                    args.source)
        plot_gain_decomposition(gains, carry, out_dir / "gain_decomposition.png", args.source)
        plot_training_curves(te, out_dir / "training_curves_comparison.png")
        plot_convergence_efficiency(te, models_metrics, out_dir / "convergence_efficiency.png",
                                    args.source)
    except Exception as exc:
        print(f"  [WARN] 绘图失败 ({type(exc).__name__}: {exc}); 继续生成 HTML/JSON 报告")

    # ── 汇总 JSON ──
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "compare_fusion.py v5.5",
        "outputs_root": str(root),
        "source": args.source,
        "sig_delta": SIG_DELTA,
        "models": {
            n: {
                "label": MODELS[n]["label"], "family": MODELS[n]["family"],
                "input": MODELS[n]["input"], "dir": MODELS[n]["dir"],
                "metrics_path": str(paths[n]),
                "n_params": params[n][0], "n_params_source": params[n][1],
                "metrics": models_metrics[n],
                "train_effect": {k: v for k, v in (te[n] or {}).items() if k != "curve"},
                "config": cfg[n],
            } for n in available
        },
        "gains": [{"key": g["key"], "a": g["a"], "b": g["b"],
                   "delta_mIoU": round(g["delta"], 6),
                   "verdict": verdict(g["delta"])} for g in gains],
        "index_carry_ratio": None if carry is None else round(carry, 6),
    }
    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_dir / 'comparison_summary.json'}")

    # ── HTML ──
    print("\n--- 生成 HTML 报告 ---")
    ctx = dict(models=models_metrics, full=full, train_effects=te, params=params,
               gains=gains, carry=carry, source=args.source, root=root, out_dir=out_dir,
               cfg=cfg, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report = generate_html_report(ctx)

    # ── 控制台增益摘要 ──
    print(f"\n{'=' * 78}")
    print("增益分解 (Δ mIoU, 判定阈值 ±%.2f)" % SIG_DELTA)
    print("=" * 78)
    for g in gains:
        print(f"  {g['key']:<16s} {MODELS[g['a']]['short']:>10s} − "
              f"{MODELS[g['b']]['short']:<10s} = {g['delta']:+.4f}   {verdict(g['delta'])}")
    if carry is not None:
        print(f"  {'指数承载比':<16s} {'':>10s}   {'':<10s} = {carry * 100:+.1f}%")

    print(f"\n报告: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
