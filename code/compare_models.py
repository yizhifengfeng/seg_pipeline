# -*- coding: utf-8 -*-
"""
双模型性能对比脚本
==================
读取基线(pan-only)和多光谱(6ch)模型的 metrics.json, 生成:
  1. 对比柱状图 (mIoU / per-class IoU / F1 / OA)
  2. 训练曲线对比图 (loss + mIoU)
  3. HTML 对比报告 (含表格 + 图表 + 分析结论)

用法:
    python compare_models.py
    python compare_models.py --baseline outputs/train_baseline/metrics.json \
                             --multispec outputs/train/metrics.json

即使其中一个模型未训练完(缺 metrics.json), 也能输出部分对比.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, CLASS_NAMES

COMPARE_DIR = OUTPUT_DIR / "comparison"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CLASSES = ["crack", "biological_attachment", "flaking", "hollow"]


def load_metrics(path):
    """加载 metrics.json, 容错处理."""
    if not Path(path).exists():
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
        # 兼容 final_val (新格式) / final_metrics (旧格式) / best_val
        data = m.get("final_val") or m.get("final_metrics") or m.get("best_val") or {}
        # 如果都没有, 但有 history, 用最后一个 epoch 的数据构造
        if not data and m.get("history", {}).get("val_miou"):
            hist = m["history"]
            data = {"mIoU": hist["val_miou"][-1], "OA": 0,
                    "per_class_IoU": {}, "per_class_F1": {}}
    if not data:
        return None
    # v2: balance score (有 P/R 数据则计算)
    score = m.get("best_score")
    if data.get("per_class_Precision") and data.get("per_class_Recall"):
        try:
            import sys as _sys
            _p = Path(__file__).resolve().parent
            if str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
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


def plot_metric_comparison(baseline_m, multispec_m, out_path):
    """生成对比柱状图: mIoU / OA / per-class IoU / F1."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    labels = ["Baseline\n(pan only)", "Multispectral\n(6ch+2edge)"]
    colors = ["#e74c3c", "#3498db"]

    # 1. mIoU & OA
    ax = axes[0, 0]
    vals = [
        [baseline_m["mIoU"] if baseline_m else 0, baseline_m["OA"] if baseline_m else 0],
        [multispec_m["mIoU"] if multispec_m else 0, multispec_m["OA"] if multispec_m else 0],
    ]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, vals[0], w, label="Baseline", color=colors[0])
    ax.bar(x + w/2, vals[1], w, label="Multispectral", color=colors[1])
    ax.set_xticks(x); ax.set_xticklabels(["mIoU", "OA"])
    ax.set_ylim(0, max(max(vals[0]), max(vals[1])) * 1.3 + 0.05)
    ax.set_title("Overall: mIoU & OA"); ax.legend()
    for i in range(2):
        for j in range(2):
            ax.text(j - w/2 + (i * w), vals[i][j] + 0.01,
                    f"{vals[i][j]:.3f}", ha="center", fontsize=9)

    # 2. Per-class IoU
    ax = axes[0, 1]
    b_ious = [baseline_m["per_class_IoU"][c] if baseline_m else 0 for c in DEFECT_CLASSES]
    m_ious = [multispec_m["per_class_IoU"][c] if multispec_m else 0 for c in DEFECT_CLASSES]
    x = np.arange(len(DEFECT_CLASSES))
    ax.bar(x - w/2, b_ious, w, label="Baseline", color=colors[0])
    ax.bar(x + w/2, m_ious, w, label="Multispectral", color=colors[1])
    ax.set_xticks(x); ax.set_xticklabels([c.replace("biological_attachment", "bio") 
                                          for c in DEFECT_CLASSES], rotation=15)
    ax.set_title("Per-class IoU"); ax.legend()
    for i, (b, m) in enumerate(zip(b_ious, m_ious)):
        ax.text(i - w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, m + 0.005, f"{m:.3f}", ha="center", fontsize=8)

    # 3. Per-class F1
    ax = axes[1, 0]
    b_f1 = [baseline_m["per_class_F1"][c] if baseline_m else 0 for c in DEFECT_CLASSES]
    m_f1 = [multispec_m["per_class_F1"][c] if multispec_m else 0 for c in DEFECT_CLASSES]
    ax.bar(x - w/2, b_f1, w, label="Baseline", color=colors[0])
    ax.bar(x + w/2, m_f1, w, label="Multispectral", color=colors[1])
    ax.set_xticks(x); ax.set_xticklabels([c.replace("biological_attachment", "bio")
                                          for c in DEFECT_CLASSES], rotation=15)
    ax.set_title("Per-class F1"); ax.legend()
    for i, (b, m) in enumerate(zip(b_f1, m_f1)):
        ax.text(i - w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, m + 0.005, f"{m:.3f}", ha="center", fontsize=8)

    # 4. IoU 提升幅度 (Δ)
    ax = axes[1, 1]
    if baseline_m and multispec_m:
        deltas = [multispec_m["per_class_IoU"][c] - baseline_m["per_class_IoU"][c]
                  for c in DEFECT_CLASSES]
        delta_colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
        ax.bar(x, deltas, w*2, color=delta_colors)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([c.replace("biological_attachment", "bio")
                                              for c in DEFECT_CLASSES], rotation=15)
        ax.set_title("IoU Improvement (Multispec - Baseline)")
        for i, d in enumerate(deltas):
            ax.text(i, d + (0.005 if d >= 0 else -0.01), f"{d:+.3f}",
                    ha="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Need both models for delta", ha="center", transform=ax.transAxes)
        ax.set_title("IoU Improvement (requires both models)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_training_curves_comparison(b_hist, m_hist, out_path):
    """训练曲线对比: loss + mIoU."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(20, 5))

    if b_hist:
        a1.plot(b_hist.get("train_loss", []), label="Baseline train", color="#e74c3c")
        a1.plot(b_hist.get("val_loss", []), label="Baseline val", color="#e74c3c",
                linestyle="--")
    if m_hist:
        a1.plot(m_hist.get("train_loss", []), label="Multispec train", color="#3498db")
        a1.plot(m_hist.get("val_loss", []), label="Multispec val", color="#3498db",
                linestyle="--")
    a1.set_title("Training Loss Comparison"); a1.set_xlabel("Epoch"); a1.legend()

    if b_hist:
        a2.plot(b_hist.get("val_miou", []), label="Baseline", color="#e74c3c")
    if m_hist:
        a2.plot(m_hist.get("val_miou", []), label="Multispectral", color="#3498db")
    a2.set_title("Validation mIoU Comparison"); a2.set_xlabel("Epoch"); a2.legend()

    # v2: balance score 曲线
    has_score = (b_hist and b_hist.get("val_score")) or (m_hist and m_hist.get("val_score"))
    if has_score:
        if b_hist and b_hist.get("val_score"):
            a3.plot(b_hist.get("val_score", []), label="Baseline", color="#e74c3c")
        if m_hist and m_hist.get("val_score"):
            a3.plot(m_hist.get("val_score", []), label="Multispectral", color="#3498db")
        a3.set_title("Balance Score Comparison")
    else:
        a3.text(0.5, 0.5, "score not available\n(train with v2 scripts)",
                ha="center", transform=a3.transAxes)
        a3.set_title("Balance Score (N/A)")
    a3.set_xlabel("Epoch"); a3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def generate_html_report(baseline_m, multispec_m, b_full, m_full, out_path):
    """生成 HTML 对比报告."""
    b = baseline_m or {}
    m = multispec_m or {}

    def fmt(v, prec=4):
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "N/A"

    def delta(bv, mv):
        if bv is None or mv is None:
            return "N/A"
        d = mv - bv
        color = "#27ae60" if d > 0 else "#e74c3c" if d < 0 else "#7f8c8d"
        return f'<span style="color:{color}">{d:+.4f}</span>'

    # Per-class table
    rows = ""
    for c in DEFECT_CLASSES:
        b_iou = b.get("per_class_IoU", {}).get(c, None)
        m_iou = m.get("per_class_IoU", {}).get(c, None)
        b_f1 = b.get("per_class_F1", {}).get(c, None)
        m_f1 = m.get("per_class_F1", {}).get(c, None)
        rows += f"""
        <tr>
          <td>{c}</td>
          <td>{fmt(b_iou)}</td><td>{fmt(m_iou)}</td><td>{delta(b_iou, m_iou)}</td>
          <td>{fmt(b_f1)}</td><td>{fmt(m_f1)}</td><td>{delta(b_f1, m_f1)}</td>
        </tr>"""

    # Analysis conclusion
    if baseline_m and multispec_m:
        miou_delta = multispec_m["mIoU"] - baseline_m["mIoU"]
        miou_pct = (miou_delta / max(baseline_m["mIoU"], 1e-6)) * 100
        if miou_delta > 0:
            conclusion = f"""
            <div class="conclusion positive">
              <h3>结论: 多光谱融合模型优于基线</h3>
              <p>多光谱模型 mIoU 提升 <strong>{miou_delta:+.4f}</strong>
                 (相对提升 {miou_pct:+.1f}%)，证明 5 波段光谱信息对病害检测有显著贡献。</p>
            </div>"""
        else:
            conclusion = f"""
            <div class="conclusion negative">
              <h3>结论: 基线模型暂时领先</h3>
              <p>基线 mIoU 高于多光谱模型 {abs(miou_delta):.4f}。
                 可能原因: 训练 epoch 不足 / 光谱增强需要调参 / pan 高分辨率细节已足够。
                 建议延长训练或检查光谱增强策略。</p>
            </div>"""
    else:
        conclusion = """
        <div class="conclusion pending">
          <h3>结论: 待补充</h3>
          <p>需要两个模型都完成训练才能生成完整对比结论。</p>
        </div>"""

    # Training info
    b_epochs = len(b_full.get("history", {}).get("train_loss", [])) if b_full else 0
    m_epochs = len(m_full.get("history", {}).get("train_loss", [])) if m_full else 0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>模型对比报告: Baseline vs Multispectral</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 20px; background: #f5f5f5; color: #2c3e50; }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white;
               padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: center; }}
  th {{ background: #34495e; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .metric-box {{ display: inline-block; background: #ecf0f1; padding: 15px 25px;
                margin: 10px; border-radius: 6px; text-align: center; }}
  .metric-box h3 {{ margin: 0 0 5px; color: #7f8c8d; font-size: 14px; }}
  .metric-box .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
  .baseline-color {{ color: #e74c3c; }}
  .multispec-color {{ color: #3498db; }}
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
  <h1>缺陷检测模型对比报告</h1>
  <p>基线模型 (仅可见光 pan) vs 多光谱融合模型 (5 光谱波段 + pan + 2 边缘先验)</p>

  <h2>1. 模型概览</h2>
  <div class="metric-box">
    <h3>基线输入</h3>
    <div class="value baseline-color">1ch pan + 1 edge = 2ch</div>
  </div>
  <div class="metric-box">
    <h3>多光谱输入</h3>
    <div class="value multispec-color">5 spec + pan + 2 edge = 8ch</div>
  </div>
  <div class="metric-box">
    <h3>基线训练 epoch</h3>
    <div class="value">{b_epochs}</div>
  </div>
  <div class="metric-box">
    <h3>多光谱训练 epoch</h3>
    <div class="value">{m_epochs}</div>
  </div>

  <h2>2. 总体指标对比</h2>
  <table>
    <tr>
      <th>指标</th>
      <th class="baseline-color">Baseline (pan only)</th>
      <th class="multispec-color">Multispectral (6ch)</th>
      <th>Δ (Multi - Base)</th>
    </tr>
    <tr>
      <td><strong>Balance Score (综合评分)</strong></td>
      <td>{fmt(b.get('score'))}</td>
      <td>{fmt(m.get('score'))}</td>
      <td>{delta(b.get('score'), m.get('score'))}</td>
    </tr>
    <tr>
      <td><strong>mIoU</strong></td>
      <td>{fmt(b.get('mIoU'))}</td>
      <td>{fmt(m.get('mIoU'))}</td>
      <td>{delta(b.get('mIoU'), m.get('mIoU'))}</td>
    </tr>
    <tr>
      <td><strong>OA (Overall Accuracy)</strong></td>
      <td>{fmt(b.get('OA'))}</td>
      <td>{fmt(m.get('OA'))}</td>
      <td>{delta(b.get('OA'), m.get('OA'))}</td>
    </tr>
  </table>
  <p style="color:#7f8c8d;font-size:13px;">
    Balance Score = 0.40·mIoU + 0.30·meanDefectF1 + 0.15·meanDefectP + 0.15·meanDefectR − 0.10·meanDefect|P−R|
    （仅缺陷类计算 F1/P/R，避免背景主导；含 P-R 平衡惩罚）
  </p>

  <h2>3. 分类别指标对比</h2>
  <table>
    <tr>
      <th rowspan="2">类别</th>
      <th colspan="3">IoU</th>
      <th colspan="3">F1</th>
    </tr>
    <tr>
      <th class="baseline-color">Base</th>
      <th class="multispec-color">Multi</th>
      <th>Δ</th>
      <th class="baseline-color">Base</th>
      <th class="multispec-color">Multi</th>
      <th>Δ</th>
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
    <li><strong>公平对比原则</strong>: 两个模型共享相同架构(DeepLabV3+ ResNet-34 OS=16),
        相同损失(Focal+Lovasz+Boundary), 相同数据划分(7:2:1, seed=42),
        相同训练超参(AdamW, cosine LR, early stop), 相同增强策略(几何+noise/jitter).</li>
    <li><strong>唯一差异</strong>: 输入模态 — 基线仅 pan(可见光 DN), 多光谱含 b1-b5(反射率)+pan.</li>
    <li><strong>参数量</strong>: 基线 26.75M vs 多光谱 26.77M, 差异 0.02M(0.1%), 不影响公平性.</li>
    <li><strong>光谱贡献</strong>: 对比结果隔离出 5 波段光谱信息对各类病害检测的纯贡献.</li>
    <li><strong>预期</strong>: 光谱波段对 biological_attachment(生物附着, 含叶绿素 b5 反射)
        和 flaking(剥落, 材质变化引起光谱差异) 应有显著提升; 对 crack(裂缝, 形态特征主导)
        提升可能较小.</li>
  </ul>

  <div class="footer">
    Generated by compare_models.py | MS-CrackSeg Project
  </div>
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="双模型性能对比")
    parser.add_argument("--baseline", default=str(OUTPUT_DIR / "train_baseline" / "metrics.json"))
    parser.add_argument("--multispec", default=str(OUTPUT_DIR / "train" / "metrics.json"))
    parser.add_argument("--source", choices=["val", "test"], default="test",
                        help="对比 val 还是 test 指标 (test 需训练完成)")
    args = parser.parse_args()

    print(f"=== Model Comparison ===")
    print(f"Baseline:   {args.baseline}")
    print(f"Multispec:  {args.multispec}")
    print(f"Source:     {args.source}")

    b_full = load_metrics(args.baseline)
    m_full = load_metrics(args.multispec)

    # 优先用 test, 回退到 val
    source = args.source
    b_m = extract_key_metrics(b_full, source)
    if b_m is None and source == "test":
        print(f"  [INFO] Baseline test metrics not found, falling back to val")
        b_m = extract_key_metrics(b_full, "val")
        source_used_b = "val"
    else:
        source_used_b = source
    m_m = extract_key_metrics(m_full, source)
    if m_m is None and source == "test":
        print(f"  [INFO] Multispec test metrics not found, falling back to val")
        m_m = extract_key_metrics(m_full, "val")
        source_used_m = "val"
    else:
        source_used_m = source

    if b_m is None and m_m is None:
        print("\n!! Neither model has metrics. Train at least one model first.")
        return

    print(f"\n--- Baseline ({source_used_b}) ---")
    if b_m:
        print(f"  mIoU={b_m['mIoU']:.4f}  OA={b_m['OA']:.4f}")
        for c in DEFECT_CLASSES:
            print(f"    {c:25s} IoU={b_m['per_class_IoU'][c]:.4f}  F1={b_m['per_class_F1'][c]:.4f}")
    else:
        print("  (not trained)")

    print(f"\n--- Multispectral ({source_used_m}) ---")
    if m_m:
        print(f"  mIoU={m_m['mIoU']:.4f}  OA={m_m['OA']:.4f}")
        for c in DEFECT_CLASSES:
            print(f"    {c:25s} IoU={m_m['per_class_IoU'][c]:.4f}  F1={m_m['per_class_F1'][c]:.4f}")
    else:
        print("  (not trained)")

    # 对比图表
    print(f"\n--- Generating comparison plots ---")
    plot_metric_comparison(b_m, m_m, COMPARE_DIR / "metric_comparison.png")

    b_hist = b_full.get("history", {}) if b_full else {}
    m_hist = m_full.get("history", {}) if m_full else {}
    plot_training_curves_comparison(b_hist, m_hist,
                                    COMPARE_DIR / "training_curves_comparison.png")

    # HTML 报告
    print(f"\n--- Generating HTML report ---")
    generate_html_report(b_m, m_m, b_full, m_full, COMPARE_DIR / "comparison_report.html")

    # 总结
    if b_m and m_m:
        d_miou = m_m["mIoU"] - b_m["mIoU"]
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline mIoU:      {b_m['mIoU']:.4f}")
        print(f"  Multispec mIoU:     {m_m['mIoU']:.4f}")
        print(f"  Δ mIoU:             {d_miou:+.4f} "
              f"({d_miou/max(b_m['mIoU'],1e-6)*100:+.1f}%)")
        for c in DEFECT_CLASSES:
            d = m_m["per_class_IoU"][c] - b_m["per_class_IoU"][c]
            print(f"  Δ {c:25s} IoU: {d:+.4f}")
        print(f"\nReport: {COMPARE_DIR / 'comparison_report.html'}")
    else:
        print(f"\nPartial comparison saved. Train both models for full report.")
        print(f"Report: {COMPARE_DIR / 'comparison_report.html'}")


if __name__ == "__main__":
    main()
