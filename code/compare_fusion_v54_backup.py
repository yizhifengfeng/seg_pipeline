# -*- coding: utf-8 -*-
"""
多模型对比脚本 (方案A + 阶段二指数增强 E2/E3)
======================================
对比四个模型 (均已 v4 配置训练):
  1. pan 基线      (1ch pan + 1 edge)          -> outputs/train_baseline/metrics.json
  2. RGB 基线      (3ch b1/b2/b3 + 1 edge)     -> outputs/train_rgb/metrics.json
  3. 早期融合 6ch  (5 spec + pan + 2 edge)     -> outputs/train/metrics.json
  4. 中期融合      (双塔特征级融合)            -> outputs/train_intermediate/metrics.json

核心产出 (方案A 的 2×2 增益分解):
  - 光谱增益     Δ_spectral = ms(6ch) - RGB(3ch)   [多光谱 vs 可见光彩色]
  - 波段数增益   Δ_color    = RGB(3ch) - pan(1ch)   [可见光内部分谱 vs 全谱灰度]
  - 总增益       Δ_total    = ms(6ch) - pan(1ch)    [已有的总增益]

任何一个模型缺失 metrics.json 时自动退化为可用模型数的对比, 不报错.

用法:
    python compare_fusion.py --source test
    python compare_fusion.py --source val
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, CLASS_NAMES


def _setup_cjk_font():
    """配置 matplotlib 中文字体, 避免图表中文乱码 (Windows: SimHei / YaHei)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        for name in ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"):
            for f in font_manager.fontManager.ttflist:
                if f.name == name:
                    matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return
    except Exception:
        pass

COMPARE_DIR = OUTPUT_DIR / "comparison"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CLASSES = ["crack", "biological_attachment", "flaking"]

# 六模型定义: 名称 -> (metrics 相对路径, 显示名, 颜色, 输入说明)
MODELS = {
    "pan":       ("train_baseline",       "Pan 基线 (1ch)",      "#e74c3c", "1ch pan + 1 edge = 2ch"),
    "rgb":       ("train_rgb",            "RGB 基线 (3ch)",      "#f39c12", "3ch b1/b2/b3 + 1 edge = 4ch"),
    "early":     ("train",                "早期融合 (6ch)",      "#3498db", "5 spec + pan + 2 edge = 8ch"),
    "intermediate": ("train_intermediate", "中期融合 (双塔)",     "#9b59b6", "双塔特征级融合"),
    "ms_indices":   ("train_indices_ms",   "早期+指数 (10ch)",    "#16a085", "6ch base + 4 NDI + 2 edge = 12ch"),
    "rgb_indices":  ("train_indices_rgb",  "RGB+指数 (7ch)",      "#d35400", "3ch RGB + 4 NDI + 1 edge = 8ch"),
}


def load_metrics(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_key_metrics(m, source="test"):
    """提取关键指标, 与 compare_models.extract_key_metrics 语义一致."""
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


def plot_overall(models_metrics, out_path):
    """总体指标柱状图 (mIoU / OA / score)."""
    _setup_cjk_font()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k in MODELS if k in models_metrics]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key in zip(axes, ["mIoU", "OA", "score"]):
        vals = [models_metrics[n][key] if models_metrics[n] else 0 for n in names]
        colors = [MODELS[n][2] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([MODELS[n][1] for n in names], rotation=20, ha="right", fontsize=8)
        ax.set_title(key)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class(models_metrics, out_path):
    """分类别 IoU 分组柱状图."""
    _setup_cjk_font()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k in MODELS if models_metrics.get(k)]
    if not names:
        return
    x = np.arange(len(DEFECT_CLASSES))
    n = len(names)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(names):
        vals = [models_metrics[name]["per_class_IoU"][c] for c in DEFECT_CLASSES]
        ax.bar(x + (i - n / 2 + 0.5) * w, vals, w, label=MODELS[name][1], color=MODELS[name][2])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("biological_attachment", "bio") for c in DEFECT_CLASSES])
    ax.set_ylabel("IoU"); ax.set_title("Per-class IoU (六模型)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_gain_decomposition(models_metrics, out_path):
    """2×2 增益分解图 (方案A 核心)."""
    _setup_cjk_font()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pan = models_metrics.get("pan")
    rgb = models_metrics.get("rgb")
    early = models_metrics.get("early")
    ms_idx = models_metrics.get("ms_indices")
    rgb_idx = models_metrics.get("rgb_indices")

    fig, ax = plt.subplots(figsize=(9, 7))
    rows = []
    if pan and rgb:
        rows.append(("Δ_color (RGB − pan)\n可见光内: 分谱 vs 全谱",
                     [rgb["mIoU"] - pan["mIoU"]]))
    if rgb and early:
        rows.append(("Δ_spectral (early − RGB)\n光谱增益: 多光谱 vs 可见光",
                     [early["mIoU"] - rgb["mIoU"]]))
    if pan and early:
        rows.append(("Δ_total (early − pan)\n总增益", [early["mIoU"] - pan["mIoU"]]))
    # 阶段二新增: 指数增益分解 (E2/E3)
    if early and ms_idx:
        rows.append(("Δ_index (ms_idx − early)\n指数增益: 显式指数通道 vs 原始波段",
                     [ms_idx["mIoU"] - early["mIoU"]]))
    if rgb and rgb_idx:
        rows.append(("Δ_index_rgb (rgb_idx − RGB)\n可见光上叠加指数的增益",
                     [rgb_idx["mIoU"] - rgb["mIoU"]]))

    if not rows:
        ax.text(0.5, 0.5, "需要至少两个模型才能做增益分解", ha="center", transform=ax.transAxes)
        ax.set_title("增益分解 (需 pan/rgb/early)")
    else:
        for i, (label, delta) in enumerate(rows):
            color = "#27ae60" if delta[0] > 0 else "#e74c3c"
            ax.barh(i, delta[0], color=color)
            ax.text(delta[0] + (0.003 if delta[0] >= 0 else -0.003), i,
                    f"{delta[0]:+.4f}", va="center", ha="left" if delta[0] >= 0 else "right",
                    fontsize=10)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ mIoU")
        ax.set_title("增益分解 (方案A): 彩色增益 vs 光谱增益")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def generate_html_report(models_metrics, full_metrics, out_path):
    """多模型 HTML 对比报告."""
    def fmt(v, prec=4):
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "N/A"

    def delta(a, b):
        if a is None or b is None:
            return "N/A"
        d = b - a
        color = "#27ae60" if d > 0 else "#e74c3c" if d < 0 else "#7f8c8d"
        return f'<span style="color:{color}">{d:+.4f}</span>'

    # 总体指标表
    names = [k for k in MODELS if k in models_metrics]
    overall_rows = ""
    for key, label in [("score", "Balance Score"), ("mIoU", "mIoU"), ("OA", "OA")]:
        cells = "".join(f"<td>{fmt(models_metrics[n][key]) if models_metrics[n] else 'N/A'}</td>"
                        for n in MODELS if n in names)
        overall_rows += f"<tr><td><strong>{label}</strong></td>{cells}</tr>"

    # 分类别 IoU 表
    per_class_rows = ""
    for c in DEFECT_CLASSES:
        cells = "".join(
            f"<td>{fmt(models_metrics[n]['per_class_IoU'][c]) if models_metrics[n] else 'N/A'}</td>"
            for n in MODELS if n in names)
        per_class_rows += f"<tr><td>{c}</td>{cells}</tr>"

    # 增益分解表
    gain_rows = ""
    pan, rgb, early = (models_metrics.get(k) for k in ("pan", "rgb", "early"))
    ms_idx = models_metrics.get("ms_indices")
    rgb_idx = models_metrics.get("rgb_indices")
    if pan and rgb:
        d = rgb["mIoU"] - pan["mIoU"]
        gain_rows += (f"<tr><td>彩色增益 (RGB − pan)</td><td>{fmt(d)}</td>"
                      f"<td>{'分谱 vs 全谱灰度' if True else ''}</td></tr>")
    if rgb and early:
        d = early["mIoU"] - rgb["mIoU"]
        gain_rows += (f"<tr><td>光谱增益 (early − RGB)</td><td>{fmt(d)}</td>"
                      f"<td>NIR/RedEdge 贡献</td></tr>")
    if pan and early:
        d = early["mIoU"] - pan["mIoU"]
        gain_rows += (f"<tr><td>总增益 (early − pan)</td><td>{fmt(d)}</td>"
                      f"<td>光谱 + 彩色</td></tr>")
    if early and ms_idx:
        d = ms_idx["mIoU"] - early["mIoU"]
        gain_rows += (f"<tr><td>指数增益 (ms_idx − early)</td><td>{fmt(d)}</td>"
                      f"<td>5 个 NDI 显式通道 vs 原始波段的额外增益 (E2)</td></tr>")
    if rgb and rgb_idx:
        d = rgb_idx["mIoU"] - rgb["mIoU"]
        gain_rows += (f"<tr><td>可见光指数增益 (rgb_idx − RGB)</td><td>{fmt(d)}</td>"
                      f"<td>RGB 上叠加 5 个 NDI 的增益 (E3)</td></tr>")
    if rgb and rgb_idx and early and abs(early["mIoU"] - rgb["mIoU"]) > 1e-6:
        carry = (rgb_idx["mIoU"] - rgb["mIoU"]) / (early["mIoU"] - rgb["mIoU"])
        gain_rows += (f"<tr><td>光谱增益的指数承载比</td><td>{carry*100:.1f}%</td>"
                      f"<td>Δ_index_rgb / Δ_spectral: 光谱增益中可被 4 个 NDI 承载的比例</td></tr>")

    header_cells = "".join(f'<th style="color:{MODELS[n][2]}">{MODELS[n][1]}</th>'
                           for n in MODELS if n in names)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>多模型对比报告 (方案A + E2/E3)</title>
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
  .img-container {{ text-align: center; margin: 20px 0; }}
  .img-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #eee;
            color: #95a5a6; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>多模型对比报告 (方案A + 阶段二指数增强)</h1>
  <p>pan 基线 / RGB 基线 / 早期融合 / 中期融合 / 早期+指数 / RGB+指数 (同数据同超参)</p>

  <h2>1. 模型概览</h2>
  <table>
    <tr><th>模型</th><th>输入</th><th>训练 epoch</th></tr>
    {''.join(f"<tr><td>{MODELS[n][1]}</td><td>{MODELS[n][3]}</td><td>"
             f"{(lambda fm: len(fm['history'].get('train_loss', [])) if fm and 'history' in fm else 'N/A')((full_metrics.get(n)))}</td></tr>"
             for n in names)}
  </table>

  <h2>2. 总体指标</h2>
  <table>
    <tr><th>指标</th>{header_cells}</tr>
    {overall_rows}
  </table>

  <h2>3. 分类别 IoU</h2>
  <table>
    <tr><th>类别</th>{header_cells}</tr>
    {per_class_rows}
  </table>

  <h2>4. 增益分解 (方案A 核心)</h2>
  <table>
    <tr><th>增益项</th><th>Δ mIoU</th><th>含义</th></tr>
    {gain_rows}
  </table>
  <div class="img-container"><img src="gain_decomposition.png" alt="增益分解"></div>

  <h2>5. 对比图表</h2>
  <div class="img-container"><img src="overall_comparison.png" alt="总体指标"></div>
  <div class="img-container"><img src="per_class_comparison.png" alt="分类别指标"></div>

  <h2>6. 科学结论框架</h2>
  <ul>
    <li><strong>光谱增益</strong> = 早期融合(6ch) − RGB(3ch): 隔离 NIR/RedEdge 的纯光谱贡献.</li>
    <li><strong>彩色增益</strong> = RGB(3ch) − pan(1ch): 可见光内部"分谱采样"相对"全谱灰度"的增益.</li>
    <li><strong>总增益</strong> = 早期融合(6ch) − pan(1ch): 光谱 + 彩色的叠加.</li>
    <li><strong>指数增益</strong> = ms_idx(10ch) − early(6ch): 4 个 NDI 显式通道在原始波段之上的额外增益 (E2).</li>
    <li><strong>指数承载比</strong> = (rgb_idx − RGB) / (early − RGB): 光谱增益中可被 4 个 NDI 承载的比例 (E3).</li>
    <li><strong>客观性红线</strong>: 无论增益正负都如实报告; &lt;2 mIoU 点时注明"无显著差异"; 单 seed 结论注明局限.</li>
  </ul>

  <div class="footer">Generated by compare_fusion.py | MS-CrackSeg Project</div>
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="多模型对比 (方案A + E2/E3)")
    parser.add_argument("--source", choices=["val", "test"], default="test")
    parser.add_argument("--pan", default=None)
    parser.add_argument("--rgb", default=None)
    parser.add_argument("--early", default=None)
    parser.add_argument("--intermediate", default=None)
    parser.add_argument("--ms-indices", default=None, dest="ms_indices")
    parser.add_argument("--rgb-indices", default=None, dest="rgb_indices")
    args = parser.parse_args()

    paths = {
        "pan": args.pan or str(OUTPUT_DIR / "train_baseline" / "metrics.json"),
        "rgb": args.rgb or str(OUTPUT_DIR / "train_rgb" / "metrics.json"),
        "early": args.early or str(OUTPUT_DIR / "train" / "metrics.json"),
        "intermediate": args.intermediate or str(OUTPUT_DIR / "train_intermediate" / "metrics.json"),
        "ms_indices": args.ms_indices or str(OUTPUT_DIR / "train_indices_ms" / "metrics.json"),
        "rgb_indices": args.rgb_indices or str(OUTPUT_DIR / "train_indices_rgb" / "metrics.json"),
    }

    full = {}
    models_metrics = {}
    for name in MODELS:
        m = load_metrics(paths[name])
        full[name] = m
        mm = extract_key_metrics(m, args.source)
        if mm is None and args.source == "test":
            mm = extract_key_metrics(m, "val")
        models_metrics[name] = mm
        status = "OK" if mm else "缺失"
        print(f"  [{name:12s}] {paths[name]}  ->  {status}")

    available = [n for n in MODELS if models_metrics[n]]
    if not available:
        print("\n!! 没有任何模型完成训练. 先跑 train*.py.")
        return

    print(f"\n可用模型: {[MODELS[n][1] for n in available]}")
    for n in available:
        mm = models_metrics[n]
        print(f"  {MODELS[n][1]:20s} mIoU={mm['mIoU']:.4f} OA={mm['OA']:.4f} "
              f"score={mm['score']:.4f}")

    # 生成图表
    plot_overall(models_metrics, COMPARE_DIR / "overall_comparison.png")
    plot_per_class(models_metrics, COMPARE_DIR / "per_class_comparison.png")
    plot_gain_decomposition(models_metrics, COMPARE_DIR / "gain_decomposition.png")
    generate_html_report(models_metrics, full, COMPARE_DIR / "fusion_comparison_report.html")

    # 控制台增益分解摘要
    pan, rgb, early = (models_metrics.get(k) for k in ("pan", "rgb", "early"))
    ms_idx = models_metrics.get("ms_indices")
    rgb_idx = models_metrics.get("rgb_indices")
    print(f"\n{'='*60}")
    print("增益分解 (方案A + 阶段二指数增益)")
    print(f"{'='*60}")
    if pan and rgb:
        print(f"  彩色增益 (RGB − pan):     {rgb['mIoU'] - pan['mIoU']:+.4f}")
    if rgb and early:
        print(f"  光谱增益 (early − RGB):   {early['mIoU'] - rgb['mIoU']:+.4f}")
    if pan and early:
        print(f"  总增益   (early − pan):   {early['mIoU'] - pan['mIoU']:+.4f}")
    if early and ms_idx:
        print(f"  指数增益 (ms_idx − early): {ms_idx['mIoU'] - early['mIoU']:+.4f}  "
              f"[E2: 原始波段之上指数还有增益吗]")
    if rgb and rgb_idx:
        print(f"  可见光指数增益 (rgb_idx − RGB): {rgb_idx['mIoU'] - rgb['mIoU']:+.4f}  "
              f"[E3: RGB+指数逼近多光谱多少]")
    if rgb and rgb_idx and early and abs(early["mIoU"] - rgb["mIoU"]) > 1e-6:
        carry = (rgb_idx["mIoU"] - rgb["mIoU"]) / (early["mIoU"] - rgb["mIoU"])
        print(f"  光谱增益的指数承载比:     {carry*100:.1f}%  "
              f"[Δ_index_rgb / Δ_spectral]")
    print(f"\nReport: {COMPARE_DIR / 'fusion_comparison_report.html'}")


if __name__ == "__main__":
    main()
