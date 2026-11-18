# -*- coding: utf-8 -*-
"""
多模型性能对比脚本 (v7: 支持五模型, 含中期窄版消融)
====================================
支持最多四个模型的 metrics.json 对比:
  1. RGB 基线        (train_rgb)          — 可见光 RGB 3ch
  2. pan 基线        (train_baseline)     — 全色 pan 2ch
  3. 早期融合        (train)              — 像素级 6ch+2edge=8ch
  4. 中期融合        (train_intermediate) — 特征级双塔

生成:
  1. 对比柱状图 (mIoU / per-class IoU / F1 / Δ)
  2. 训练曲线对比图 (loss + mIoU + score)
  3. HTML 对比报告 (四列表格 + 图表 + 融合策略结论)

用法:
    python compare_models.py                          # 自动检测全部模型
    python compare_models.py --rgb a.json --baseline b.json \\
        --multispec c.json --intermediate d.json --out_dir ../outputV5/comparison

即使其中某些模型未训练完(缺 metrics.json), 也能输出部分对比.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, CLASS_NAMES

DEFECT_CLASSES = ["crack", "biological_attachment", "flaking"]

# 五模型配色与标签
COLORS = {"rgb": "#e67e22", "baseline": "#e74c3c",
          "multispec": "#3498db", "intermediate": "#8e44ad",
          "narrow": "#16a085"}
LABELS = {"rgb": "RGB Baseline\n(3ch)",
          "baseline": "Pan Baseline\n(2ch)",
          "multispec": "Early Fusion\n(8ch)",
          "intermediate": "Intermediate\n(dual-tower)",
          "narrow": "Intermediate\nNarrow (0.5x)"}
TITLE = {"rgb": "RGB 基线", "baseline": "pan 基线",
         "multispec": "早期融合", "intermediate": "中期满宽",
         "narrow": "中期窄版"}


def load_metrics(path):
    """加载 metrics.json, 容错处理."""
    if not path or not Path(path).exists():
        print(f"  [WARN] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def extract_key_metrics(m, source="val"):
    """提取关键指标. source: 'val' or 'test'.
    兼容新旧格式: final_val (新) / final_metrics (旧) / best_val.
    v2: 若 metrics 含 balance score 原始数据则计算, 否则用 best_score 字段.
    """
    if m is None:
        return None
    if source == "test" and m.get("test"):
        data = m["test"]
    else:
        data = m.get("final_val") or m.get("final_metrics") or m.get("best_val") or {}
        if not data and m.get("history", {}).get("val_miou"):
            hist = m["history"]
            data = {"mIoU": hist["val_miou"][-1], "OA": 0,
                    "per_class_IoU": {}, "per_class_F1": {}}
    if not data:
        return None
    score = m.get("best_score")
    if data.get("per_class_Precision") and data.get("per_class_Recall"):
        try:
            _p = Path(__file__).resolve().parent
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            from train import compute_balance_score as _cbs
            score = _cbs(data)
        except Exception:
            pass
    return {
        "mIoU": data.get("mIoU", 0),
        "OA": data.get("OA", 0),
        "score": score,
        "per_class_IoU": {c: data.get("per_class_IoU", {}).get(c, 0) for c in DEFECT_CLASSES},
        "per_class_F1": {c: data.get("per_class_F1", {}).get(c, 0) for c in DEFECT_CLASSES},
        "per_class_Precision": {c: data.get("per_class_Precision", {}).get(c, 0) for c in DEFECT_CLASSES},
        "per_class_Recall": {c: data.get("per_class_Recall", {}).get(c, 0) for c in DEFECT_CLASSES},
    }


def plot_metric_comparison(models, out_path):
    """生成对比柱状图: mIoU / OA / per-class IoU / F1. models: [(key, m), ...]"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    n = len(models)
    w = 0.8 / max(n, 1)

    def grouped(ax, categories, key, title):
        x = np.arange(len(categories))
        for i, (key_name, m) in enumerate(models):
            lab = LABELS[key_name].replace("\n", " ")
            vals = [m[key][c] if m else 0 for c in categories]
            ax.bar(x + (i - (n-1)/2) * w, vals, w, label=lab, color=COLORS[key_name])
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - (n-1)/2) * w, v + 0.005, f"{v:.3f}",
                        ha="center", fontsize=7 if n > 2 else 8)
        disp = [c.replace("biological_attachment", "bio") for c in categories]
        ax.set_xticks(x); ax.set_xticklabels(disp, rotation=15)
        ax.set_title(title); ax.legend(fontsize=8)

    # 1. mIoU & OA
    ax = axes[0, 0]
    for i, (key_name, m) in enumerate(models):
        vals = [m["mIoU"] if m else 0, m["OA"] if m else 0]
        ax.bar(np.arange(2) + (i - (n-1)/2) * w, vals, w,
               label=LABELS[key_name].replace("\n", " "), color=COLORS[key_name])
        for j, v in enumerate(vals):
            ax.text(j + (i - (n-1)/2) * w, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(["mIoU", "OA"])
    top = max((m["mIoU"] for _, m in models if m), default=0)
    ax.set_ylim(0, top * 1.25 + 0.05)
    ax.set_title("Overall: mIoU & OA"); ax.legend(fontsize=8)

    # 2. Per-class IoU
    grouped(axes[0, 1], DEFECT_CLASSES, "per_class_IoU", "Per-class IoU")
    # 3. Per-class F1
    grouped(axes[1, 0], DEFECT_CLASSES, "per_class_F1", "Per-class F1")

    # 4. Δ IoU: 各融合模型 vs 最强单模态基线 (RGB 若有, 否则 pan)
    ax = axes[1, 1]
    ref_key, ref_m = None, None
    for k in ("rgb", "baseline"):
        for mk, mm in models:
            if mk == k and mm:
                ref_key, ref_m = k, mm
                break
        if ref_m:
            break
    if ref_m:
        refs = [(mk, mm) for mk, mm in models
                if mk in ("multispec", "intermediate", "narrow") and mm]
        nrefs = len(refs)
        x = np.arange(len(DEFECT_CLASSES))
        wr = 0.8 / max(nrefs, 1)
        for i, (mk, mm) in enumerate(refs):
            deltas = [mm["per_class_IoU"][c] - ref_m["per_class_IoU"][c]
                      for c in DEFECT_CLASSES]
            bars = ax.bar(x + (i - (nrefs-1)/2) * wr, deltas, wr,
                          label=f"{TITLE[mk]} - {TITLE[ref_key]}",
                          color=COLORS[mk], alpha=0.85)
            for bar, d in zip(bars, deltas):
                ax.text(bar.get_x() + bar.get_width()/2,
                        d + (0.005 if d >= 0 else -0.012), f"{d:+.3f}",
                        ha="center", fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("biological_attachment", "bio")
                            for c in DEFECT_CLASSES], rotation=15)
        ax.set_title(f"IoU Improvement (vs {TITLE[ref_key]})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Need a baseline for delta", ha="center",
                transform=ax.transAxes)
        ax.set_title("IoU Improvement (requires baseline)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_training_curves_comparison(hists, out_path):
    """训练曲线对比: loss + mIoU + score. hists: [(key, hist_dict), ...]"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(20, 5))

    for key, h in hists:
        if not h:
            continue
        lab = TITLE[key]
        col = COLORS[key]
        a1.plot(h.get("train_loss", []), label=f"{lab} train", color=col)
        a1.plot(h.get("val_loss", []), label=f"{lab} val", color=col, linestyle="--")
        a2.plot(h.get("val_miou", []), label=lab, color=col)
    a1.set_title("Training Loss Comparison"); a1.set_xlabel("Epoch"); a1.legend(fontsize=8)
    a2.set_title("Validation mIoU Comparison"); a2.set_xlabel("Epoch"); a2.legend(fontsize=8)

    has_score = any(h and h.get("val_score") for _, h in hists)
    if has_score:
        for key, h in hists:
            if h and h.get("val_score"):
                a3.plot(h.get("val_score", []), label=TITLE[key], color=COLORS[key])
        a3.set_title("Balance Score Comparison")
    else:
        a3.text(0.5, 0.5, "score not available\n(train with v2 scripts)",
                ha="center", transform=a3.transAxes)
        a3.set_title("Balance Score (N/A)")
    a3.set_xlabel("Epoch"); a3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def generate_html_report(models, fulls, out_path):
    """生成 HTML 多模型对比报告. models: [(key, m)], fulls: [(key, full)]"""
    mm = {k: (m or {}) for k, m in models}

    def fmt(v, prec=4):
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "N/A"

    def delta(bv, mv):
        if bv is None or mv is None:
            return "N/A"
        d = mv - bv
        color = "#27ae60" if d > 0 else "#e74c3c" if d < 0 else "#7f8c8d"
        return f'<span style="color:{color}">{d:+.4f}</span>'

    def pct(bv, mv):
        if not bv:
            return "N/A"
        return f'{(mv - bv) / bv * 100:+.1f}%'

    g = lambda k, d, kk: (d.get(kk, {}).get(c) if d else None)

    # 表头: 存在的模型列
    present = [k for k, m in models if m is not None]
    th_cls = {"rgb": "rgb-color", "baseline": "baseline-color",
              "multispec": "multispec-color", "intermediate": "inter-color",
              "narrow": "narrow-color"}

    # Per-class 表: 每个存在的模型一列 IoU + 一列 F1
    head_cells = "".join(
        f'<th class="{th_cls[k]}">{TITLE[k]}</th>' for k in present)
    rows = ""
    for c in DEFECT_CLASSES:
        iou_cells = "".join(
            f'<td>{fmt(mm[k].get("per_class_IoU", {}).get(c))}</td>' for k in present)
        f1_cells = "".join(
            f'<td>{fmt(mm[k].get("per_class_F1", {}).get(c))}</td>' for k in present)
        rows += f"<tr><td>{c}</td>{iou_cells}{f1_cells}</tr>"

    # 总体表
    overall_rows = ""
    for label, key in [("Balance Score (综合评分)", "score"),
                       ("mIoU", "mIoU"), ("OA (Overall Accuracy)", "OA")]:
        cells = "".join(f"<td>{fmt(mm[k].get(key))}</td>" for k in present)
        overall_rows += f"<tr><td><strong>{label}</strong></td>{cells}</tr>"

    # 结论段
    r, b, e, it = mm.get("rgb"), mm.get("baseline"), mm.get("multispec"), mm.get("intermediate")
    nw = mm.get("narrow")
    parts = []
    if r and b:
        parts.append(
            f'<p>① <strong>模态基线（RGB vs pan）</strong>: mIoU {fmt(r.get("mIoU"))} vs {fmt(b.get("mIoU"))}'
            f'（{pct(b.get("mIoU"), r.get("mIoU"))}）。RGB 是更强的单模态基线，'
            f'光谱增益应以 RGB 为对照口径。</p>')
    if r and nw:
        parts.append(
            f'<p>② <strong>光谱增益（最优融合 vs RGB）</strong>: {TITLE["narrow"]} mIoU {fmt(nw.get("mIoU"))} vs RGB {fmt(r.get("mIoU"))}'
            f'（{pct(r.get("mIoU"), nw.get("mIoU"))}）。'
            + (f'分类别: crack {pct(r["per_class_IoU"]["crack"], nw["per_class_IoU"]["crack"])}, '
               f'bio {pct(r["per_class_IoU"]["biological_attachment"], nw["per_class_IoU"]["biological_attachment"])}, '
               f'flaking {pct(r["per_class_IoU"]["flaking"], nw["per_class_IoU"]["flaking"])}。'
               if nw.get("per_class_IoU") and r.get("per_class_IoU") else '')
            + '</p>')
    if e and it and nw:
        parts.append(
            f'<p>③ <strong>融合层级与容量消融</strong>: 早期 {fmt(e.get("mIoU"))} / '
            f'中期满宽(48.75M) {fmt(it.get("mIoU"))} / 中期窄版(32.6M) {fmt(nw.get("mIoU"))}。'
            + ('窄版反超满宽与早期融合 → 双塔劣势主因是满宽光谱塔的优化负担而非特征级融合本身；'
               '若窄版未超早期融合，则像素级融合仍为最优。'
               if (nw.get("mIoU") or 0) > (e.get("mIoU") or 0)
               else '窄版未超早期融合，像素级融合仍为最优方案。')
            + '</p>')
    cands = [(TITLE[k], mm[k]) for k in present]
    best_name, best_m = max(cands, key=lambda kv: kv[1].get("mIoU", 0) or 0)
    if best_m:
        parts.append(f'<p>④ <strong>最优模型</strong>: {best_name}'
                     f'（mIoU {fmt(best_m.get("mIoU"))}，OA {fmt(best_m.get("OA"))}）。</p>')
    conclusion = f"""
        <div class="conclusion positive">
          <h3>结论: 多模型对照</h3>
          {''.join(parts)}
        </div>""" if parts else """
        <div class="conclusion pending">
          <h3>结论: 待补充</h3>
          <p>需要至少两个模型完成训练才能生成完整对比结论。</p>
        </div>"""

    # 模型概览卡
    def ep(key):
        full = dict(fulls).get(key)
        return len(full.get("history", {}).get("train_loss", [])) if full else 0

    box_desc = {"rgb": ('RGB 3ch + 2edge = 5ch', 'rgb-color'),
                "baseline": ('1ch pan + 1 edge = 2ch', 'baseline-color'),
                "multispec": ('5 spec + pan + 2 edge = 8ch', 'multispec-color'),
                "intermediate": ('双塔满宽: spec 6ch + vis 2ch<br>48.75M', 'inter-color'),
                "narrow": ('双塔窄版: spec 塔通道减半<br>32.6M', 'narrow-color')}
    boxes = ""
    for k in present:
        desc, cls = box_desc[k]
        boxes += f"""
  <div class="metric-box">
    <h3>{TITLE[k]}输入</h3>
    <div class="value {cls}">{desc}<br>{ep(k)} epoch</div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>五模型对比报告: RGB / Pan / Early / Intermediate / Narrow</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 20px; background: #f5f5f5; color: #2c3e50; }}
  .container {{ max-width: 1300px; margin: 0 auto; background: white;
               padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: center; }}
  th {{ background: #34495e; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .metric-box {{ display: inline-block; background: #ecf0f1; padding: 15px 25px;
                margin: 10px; border-radius: 6px; text-align: center; vertical-align: top; }}
  .metric-box h3 {{ margin: 0 0 5px; color: #7f8c8d; font-size: 14px; }}
  .metric-box .value {{ font-size: 16px; font-weight: bold; color: #2c3e50; }}
  .rgb-color {{ color: #e67e22; }}
  .baseline-color {{ color: #e74c3c; }}
  .multispec-color {{ color: #3498db; }}
  .inter-color {{ color: #8e44ad; }}
  .narrow-color {{ color: #16a085; }}
  .conclusion {{ padding: 20px; border-radius: 6px; margin: 20px 0; }}
  .conclusion.positive {{ background: #d5edda; border-left: 4px solid #27ae60; }}
  .conclusion.negative {{ background: #f8d7da; border-left: 4px solid #e74c3c; }}
  .conclusion.pending {{ background: #fff3cd; border-left: 4px solid #f39c12; }}
  .img-container {{ text-align: center; margin: 20px 0; }}
  .img-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #eee;
            color: #95a5a6; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>缺陷检测五模型对比报告</h1>
  <p>RGB 基线 vs pan 基线 vs 早期融合（像素级）vs 中期满宽（特征级双塔 48.75M）vs 中期窄版（光谱塔减半 32.6M）</p>

  <h2>1. 模型概览</h2>
  {boxes}

  <h2>2. 总体指标对比</h2>
  <table>
    <tr>
      <th>指标</th>
      {"".join(f'<th class="{th_cls[k]}">{TITLE[k]}</th>' for k in present)}
    </tr>
    {overall_rows}
  </table>
  <p style="color:#7f8c8d;font-size:13px;">
    Balance Score = 0.40·mIoU + 0.30·meanDefectF1 + 0.15·meanDefectP + 0.15·meanDefectR − 0.10·meanDefect|P−R|
    （仅缺陷类计算 F1/P/R，避免背景主导；含 P-R 平衡惩罚）
  </p>

  <h2>3. 分类别指标对比 (IoU 与 F1)</h2>
  <table>
    <tr>
      <th rowspan="2">类别</th>
      <th colspan="{len(present)}">IoU</th>
      <th colspan="{len(present)}">F1</th>
    </tr>
    <tr>
      {"".join(f'<th class="{th_cls[k]}">{TITLE[k]}</th>' for k in present)}
      {"".join(f'<th class="{th_cls[k]}">{TITLE[k]}</th>' for k in present)}
    </tr>
    {rows}
  </table>

  <h2>4. 对比图表</h2>
  <div class="img-container">
    <img src="metric_comparison.png" alt="指标对比柱状图">
  </div>
  <div class="img-container">
    <img src="training_curves_comparison.png" alt="训练曲线对比">
  </div>

  {conclusion}

  <h2>5. 实验设置说明</h2>
  <ul>
    <li><strong>公平对比原则</strong>: 各模型共享相同数据划分、损失(Focal+Lovasz+Boundary)、
        优化器/调度(AdamW, cosine LR, early stop)与增强策略(几何+noise/jitter).</li>
    <li><strong>差异变量</strong>: 输入模态与融合层级 — RGB/pan 为单模态基线;
        早期融合在输入端像素级拼接 6ch; 中期融合为对称双塔 ResNet-34 逐级特征融合
        (参数量约 1.8×, 见 INTERMEDIATE_FUSION_PLAN.md).</li>
    <li><strong>口径说明</strong>: 光谱净增益 = 早期融合 vs <u>RGB 基线</u>（最强单模态对照）;
        vs pan 基线的增益包含可见光信息量差异，不是纯光谱贡献.</li>
    <li><strong>预期</strong>: 光谱对 bio（叶绿素 b5/NIR 反射）增益最大;
        crack 以几何形态为主，光谱增益有限; flaking 若 RGB 纹理已充分，光谱可能无净增益.</li>
  </ul>

  <div class="footer">
    Generated by compare_models.py (v6 four-model) | MS-CrackSeg Project
  </div>
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="多模型性能对比 (v7: 支持五模型)")
    parser.add_argument("--rgb", default=None,
                        help="RGB 基线 metrics.json; 不传则自动检测 "
                             "OUTPUT_DIR/train_rgb/metrics.json")
    parser.add_argument("--baseline", default=str(OUTPUT_DIR / "train_baseline" / "metrics.json"))
    parser.add_argument("--multispec", default=str(OUTPUT_DIR / "train" / "metrics.json"))
    parser.add_argument("--intermediate", default=None,
                        help="中期融合模型 metrics.json; 不传则自动检测 "
                             "OUTPUT_DIR/train_intermediate/metrics.json")
    parser.add_argument("--narrow", default=None,
                        help="中期融合窄版 (spec_width_scale=0.5) metrics.json; "
                             "不传则自动检测 OUTPUT_DIR/train_intermediate_narrow/metrics.json")
    parser.add_argument("--out_dir", default=None,
                        help="报告输出目录 (默认 OUTPUT_DIR/comparison)")
    parser.add_argument("--source", choices=["val", "test"], default="test",
                        help="对比 val 还是 test 指标 (test 需训练完成)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 自动检测 rgb / intermediate
    rgb_path = args.rgb
    if rgb_path is None:
        auto = OUTPUT_DIR / "train_rgb" / "metrics.json"
        if auto.exists():
            rgb_path = str(auto)
    inter_path = args.intermediate
    if inter_path is None:
        auto = OUTPUT_DIR / "train_intermediate" / "metrics.json"
        if auto.exists():
            inter_path = str(auto)
    narrow_path = args.narrow
    if narrow_path is None:
        auto = OUTPUT_DIR / "train_intermediate_narrow" / "metrics.json"
        if auto.exists():
            narrow_path = str(auto)

    print(f"=== Model Comparison (five-model) ===")
    print(f"RGB:           {rgb_path or '(not found)'}")
    print(f"Pan Baseline:  {args.baseline}")
    print(f"Early Fusion:  {args.multispec}")
    print(f"Intermediate:  {inter_path or '(not found)'}")
    print(f"Narrow (0.5x): {narrow_path or '(not found)'}")
    print(f"Output dir:    {out_dir}")
    print(f"Source:        {args.source}")

    fulls = {"rgb": load_metrics(rgb_path),
             "baseline": load_metrics(args.baseline),
             "multispec": load_metrics(args.multispec),
             "intermediate": load_metrics(inter_path),
             "narrow": load_metrics(narrow_path)}

    # 提取指标 (test 优先, 回退 val)
    ms, order = {}, ["rgb", "baseline", "multispec", "intermediate", "narrow"]
    for k in order:
        ms[k] = extract_key_metrics(fulls[k], args.source)
        if fulls[k] and ms[k] is None and args.source == "test":
            print(f"  [INFO] {k} test metrics not found, falling back to val")
            ms[k] = extract_key_metrics(fulls[k], "val")

    if all(v is None for v in ms.values()):
        print("\n!! No model has metrics. Train at least one model first.")
        return

    def show(name, m):
        print(f"\n--- {name} ({args.source}) ---")
        if m:
            print(f"  mIoU={m['mIoU']:.4f}  OA={m['OA']:.4f}")
            for c in DEFECT_CLASSES:
                print(f"    {c:25s} IoU={m['per_class_IoU'][c]:.4f}  F1={m['per_class_F1'][c]:.4f}")
        else:
            print("  (not trained)")

    for k in order:
        if fulls[k]:
            show(TITLE[k], ms[k])

    models = [(k, ms[k]) for k in order]
    present = [k for k in order if ms[k] is not None]

    # 对比图表
    print(f"\n--- Generating comparison plots ---")
    plot_metric_comparison(models, out_dir / "metric_comparison.png")
    hists = [(k, fulls[k].get("history", {}) if fulls[k] else {})
             for k in order if fulls[k]]
    plot_training_curves_comparison(hists, out_dir / "training_curves_comparison.png")

    # HTML 报告
    print(f"\n--- Generating HTML report ---")
    generate_html_report(models, [(k, fulls[k]) for k in order],
                         out_dir / "comparison_report.html")

    # 总结
    print(f"\n{'='*64}")
    print(f"SUMMARY ({args.source})")
    print(f"{'='*64}")
    for k in present:
        print(f"  {TITLE[k]:10s} mIoU={ms[k]['mIoU']:.4f}  OA={ms[k]['OA']:.4f}")
    # 关键 Δ
    if ms["rgb"] and ms["baseline"]:
        print(f"\n  Δ RGB - pan (模态基线强度): "
              f"{ms['rgb']['mIoU']-ms['baseline']['mIoU']:+.4f}")
    for ref in ("rgb", "baseline"):
        ref_m = ms[ref]
        if not ref_m:
            continue
        for k in ("multispec", "intermediate", "narrow"):
            if ms[k]:
                d = ms[k]["mIoU"] - ref_m["mIoU"]
                print(f"  Δ {TITLE[k]} - {TITLE[ref]}: {d:+.4f} "
                      f"({d/max(ref_m['mIoU'],1e-6)*100:+.1f}%)")
                for c in DEFECT_CLASSES:
                    dc = ms[k]["per_class_IoU"][c] - ref_m["per_class_IoU"][c]
                    print(f"      {c:25s} IoU: {dc:+.4f}")
        break  # 只用最强可用基线 (rgb 优先)
    if ms["multispec"] and ms["intermediate"]:
        print(f"  Δ 中期满宽 - 早期: {ms['intermediate']['mIoU']-ms['multispec']['mIoU']:+.4f}")
    if ms["narrow"] and ms["intermediate"]:
        print(f"  Δ 窄版 - 满宽: {ms['narrow']['mIoU']-ms['intermediate']['mIoU']:+.4f}"
              f"  (容量消融: 光谱塔减半的净效应)")
    if ms["narrow"] and ms["multispec"]:
        print(f"  Δ 窄版 - 早期: {ms['narrow']['mIoU']-ms['multispec']['mIoU']:+.4f}")
    print(f"\nReport: {out_dir / 'comparison_report.html'}")


if __name__ == "__main__":
    main()
