# -*- coding: utf-8 -*-
"""
实验台账采集器 (v1.0, 2026-11-16)
=================================================================
扫描各结果根目录下的训练记录，汇总成可无限追加的长表 ledger.csv + ledger.json，
供 experiment_board.html 看板读取。

可扩展性约定：
  - 新增一次训练（新 seed / 新批次 / 新结果根）：把结果目录放进 ROOT 下再跑本脚本即可，只加行不加列。
  - 新增模型：在 compare_fusion.MODELS 登记一行，本脚本自动纳入。
  - 新增指标：在下方 COLUMNS 与 collect() 各加一处即可。

口径复用 compare_fusion.py 的 extract_key_metrics / extract_train_effect / load_n_params，
保证台账数字与单次对比报告一致。无 test/val 指标的旧口径记录自动跳过（如 outputV4 的 5 类时代）。

用法：
  python build_ledger.py                          # 自动发现 ROOT 下所有 output* 结果根
  python build_ledger.py --roots ../outputV5 ../outputseed24
  python build_ledger.py --source test
  python build_ledger.py --out-dir ../ledger
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT
import compare_fusion as cf

DEFECT = cf.DEFECT_CLASSES

# ── CSV 长表列顺序（新增指标只在这里加列；新增模型/seed 不加列）──
COLUMNS = [
    "run_label", "model_key", "label", "family", "input", "seed", "source",
    "n_params", "metrics_path", "collected_at",
    "mIoU", "OA", "score", "defect_mean_iou",
    "IoU_bg", "IoU_crack", "IoU_bio", "IoU_flaking",
    "F1_crack", "F1_bio", "F1_flaking",
    "P_crack", "P_bio", "P_flaking",
    "R_crack", "R_bio", "R_flaking",
    "epochs", "cfg_epochs", "patience", "min_lr", "resume", "restart_epoch",
    "best_epoch", "best_val", "final_val", "decay_after_best", "tail_std",
    "train_loss_first", "train_loss_last", "stop_reason", "still_improving",
]

# 不作为 CSV 平铺列、但保留进 JSON 备查的嵌套字段
NESTED_KEYS = ("config", "per_class")


def clean(v):
    """NaN / Inf 归一为 None，避免 CSV 出现 nan/inf。"""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def discover_roots(explicit, exclude=()):
    if explicit:
        return [Path(p).resolve() for p in explicit]
    roots = []
    for d in sorted(ROOT.iterdir()):
        if not (d.is_dir() and d.name.lower().startswith("output")):
            continue
        if "comparison" in d.name.lower():
            continue
        if any(x.lower() in d.name.lower() for x in exclude):
            continue  # 旧口径 / 遗留目录
        if list(d.rglob("metrics.json")):
            roots.append(d)
    return roots


def collect(root, source):
    """单个结果根 → 记录列表。缺指标/旧口径的模型自动跳过。"""
    run_label = root.name
    records = []
    for key in cf.MODEL_ORDER:
        mdir = cf.MODELS[key]["dir"]
        mp = root / mdir / "metrics.json"
        if not mp.exists():
            continue
        m = cf.load_json(mp)
        mm = cf.extract_key_metrics(m, source)
        if mm is None and source == "test":
            mm = cf.extract_key_metrics(m, "val")
        if mm is None:
            continue  # 旧口径 / 无指标，跳过

        cfg = cf.load_json(mp.parent / "config_used.json") or {}
        te = cf.extract_train_effect(m, cfg) or {}
        n_params, _ = cf.load_n_params(key, mp)

        pci = mm["per_class_IoU_all"]
        f1 = mm["per_class_F1"]
        pr = mm["per_class_Precision"]
        rc = mm["per_class_Recall"]

        rec = {
            "run_label": run_label,
            "model_key": key,
            "label": cf.MODELS[key]["label"],
            "family": cf.MODELS[key]["family"],
            "input": cf.MODELS[key]["input"],
            "seed": clean(cfg.get("seed")),
            "source": mm["source"],
            "n_params": n_params,
            "metrics_path": str(mp),
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "mIoU": clean(mm["mIoU"]), "OA": clean(mm["OA"]),
            "score": clean(mm["score"]), "defect_mean_iou": clean(mm["defect_mean_iou"]),
            "IoU_bg": clean(pci.get("background")),
            "IoU_crack": clean(pci.get("crack")),
            "IoU_bio": clean(pci.get("biological_attachment")),
            "IoU_flaking": clean(pci.get("flaking")),
            "F1_crack": clean(f1.get("crack")),
            "F1_bio": clean(f1.get("biological_attachment")),
            "F1_flaking": clean(f1.get("flaking")),
            "P_crack": clean(pr.get("crack")),
            "P_bio": clean(pr.get("biological_attachment")),
            "P_flaking": clean(pr.get("flaking")),
            "R_crack": clean(rc.get("crack")),
            "R_bio": clean(rc.get("biological_attachment")),
            "R_flaking": clean(rc.get("flaking")),

            "epochs": clean(te.get("epochs")),
            "cfg_epochs": clean(te.get("cfg_epochs")),
            "patience": clean(te.get("patience")),
            "min_lr": clean(te.get("min_lr")),
            "resume": te.get("resume"),
            "restart_epoch": clean(te.get("restart_epoch")),
            "best_epoch": clean(te.get("best_epoch")),
            "best_val": clean(te.get("best_val")),
            "final_val": clean(te.get("final_val")),
            "decay_after_best": clean(te.get("decay_after_best")),
            "tail_std": clean(te.get("tail_std")),
            "train_loss_first": clean(te.get("train_loss_first")),
            "train_loss_last": clean(te.get("train_loss_last")),
            "stop_reason": te.get("stop_reason"),
            "still_improving": te.get("still_improving"),

            # 备查：原始配置 + 完整 per-class（不进 CSV 平铺列）
            "config": cfg,
            "per_class": {"iou": pci, "f1": f1, "precision": pr, "recall": rc},
        }
        records.append(rec)
    return records


def primary_key(rec):
    return (rec["run_label"], rec["model_key"])


def merge(old_records, new_records):
    """按主键 upsert：本轮新采覆盖旧记录；本轮没采到的历史记录保留。"""
    idx = {primary_key(r): r for r in old_records}
    for r in new_records:
        idx[primary_key(r)] = r
    return list(idx.values())


def _csv_cell(k, v):
    if v is None:
        return ""
    if k == "min_lr" and isinstance(v, float):
        # 极小学习率用定点小数，避免 Excel 按科学计数法误读
        return "{0:.10f}".format(v).rstrip("0")
    return v


def write_csv(records, csv_path):
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: _csv_cell(k, r.get(k)) for k in COLUMNS}
            w.writerow(row)


def write_json(records, roots, source, json_path):
    models_meta = {
        k: dict(label=cf.MODELS[k]["label"], family=cf.MODELS[k]["family"],
                input=cf.MODELS[k]["input"], color=cf.MODELS[k]["color"],
                dir=cf.MODELS[k]["dir"], short=cf.MODELS[k]["short"])
        for k in cf.MODEL_ORDER
    }
    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "sig_delta": cf.SIG_DELTA,
            "roots": [str(r) for r in roots],
            "columns": COLUMNS,
            "record_count": len(records),
        },
        "models_meta": models_meta,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="实验台账采集器：汇总多次训练记录为长表 + JSON")
    ap.add_argument("--roots", nargs="*", default=None,
                    help="结果根目录（多个）；缺省自动发现 ROOT 下所有 output* 目录")
    ap.add_argument("--out-dir", default=None, help="台账输出目录（缺省 ROOT/ledger）")
    ap.add_argument("--source", choices=["val", "test"], default="test")
    ap.add_argument("--exclude", nargs="*", default=["outputV4"],
                    help="排除的结果根目录名子串（默认 outputV4 旧 5 类口径）")
    ap.add_argument("--rebuild", action="store_true",
                    help="忽略旧台账全量重建（默认按主键 upsert 保留历史记录）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "ledger"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ledger.csv"
    json_path = out_dir / "ledger.json"

    roots = discover_roots(args.roots, args.exclude)
    print("=" * 78)
    print("实验台账采集器 v1.0")
    print(f"  结果根: {[r.name for r in roots]}")
    print(f"  输出目录: {out_dir}")
    print(f"  指标来源: {args.source}")
    print("=" * 78)

    new_records = []
    for r in roots:
        recs = collect(r, args.source)
        print(f"  [{r.name}] 采集 {len(recs)} 条")
        new_records.extend(recs)

    # 合并历史台账（保留本轮未采到的旧记录）；--rebuild 时全量重建
    old_records = []
    if not args.rebuild and json_path.exists():
        try:
            old_payload = json.loads(json_path.read_text(encoding="utf-8"))
            old_records = old_payload.get("records", [])
        except Exception as exc:
            print(f"  [WARN] 旧台账读取失败，按全量重建: {exc}")
            old_records = []

    records = merge(old_records, new_records)
    records.sort(key=lambda x: (x["run_label"], cf.MODEL_ORDER.index(x["model_key"])
                                if x["model_key"] in cf.MODEL_ORDER else 99))

    # 先写 JSON（看板数据源），再写 CSV；CSV 被占用时兜底写临时文件，不中断整批
    write_json(records, roots, args.source, json_path)
    csv_written = csv_path
    try:
        write_csv(records, csv_path)
    except PermissionError as exc:
        csv_written = csv_path.with_name("ledger.locked.csv")
        write_csv(records, csv_written)
        print(f"  [WARN] 原 CSV 被占用无法写入: {exc}")
        print(f"  [WARN] 已兜底写入 {csv_written}；关闭占用程序（如 WPS/Excel）后重跑即可覆盖原 CSV。")

    # 控制台摘要
    runs = sorted({r["run_label"] for r in records})
    seeds = sorted({r["seed"] for r in records if r["seed"] is not None})
    models = sorted({r["model_key"] for r in records})
    print("\n" + "-" * 78)
    print(f"  台账记录: {len(records)} 条 | 结果根: {len(runs)} | 模型: {len(models)} | seed: {seeds}")
    print(f"  CSV : {csv_written}")
    print(f"  JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
