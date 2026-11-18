# -*- coding: utf-8 -*-
"""
Dataset Restructure (本地/云端通用): img / masknpy / testimg / testmasknpy
==========================================================================
按参考数据集结构 (paper-replication/replication2/stacked_5band_train_842.py
的 stacked_5band_indices_842 数据集) 生成 npy 数据集:

  dataset_3class/
  ├── img/*.npy         训练池影像  (6, 1088, 1456) float32, 通道 [b1..b5,pan]
  ├── masknpy/*.npy     训练池 mask (1088, 1456) uint8, 值 {0:bg, 1:crack, 2:bio, 3:flaking}
  ├── testimg/*.npy     测试集影像 (同 img 格式)
  ├── testmasknpy/*.npy 测试集 mask (同 masknpy 格式)
  ├── split.json        train/val/test 场景清单 (train+val 均在 img/masknpy 中,
  │                     val 用 split.json 的 val 键区分)
  ├── norm_stats.json / norm_stats_pan.json   归一化参数副本
  └── README.md         格式说明

输入依赖 (全部来自 config.py, 环境变量适配本地/云端):
  - split.json   -> $MS_CRACKSEG_ROOT/configs/split.json
  - masks (PNG)  -> $MS_CRACKSEG_ROOT/data/masks/{category}/{stem}_mask.png
  - calibrated   -> $MS_CRACKSEG_DATA/calibrated/{category}/{stem}_b{1..5}.tif + _pan.tif

输出目录:
  $MS_CRACKSEG_DATA/dataset_3class  (可用 MS_DATASET_OUT 覆盖)
  本地: D:/deeplearning/MS-CrackSeg-main/lab/dataset_3class
  云端: /root/autodl-tmp/seg_pipeline/dataset_3class (数据盘)

用法:
  # 本地 (Git Bash):
  export MS_CRACKSEG_DATA="D:/deeplearning/MS-CrackSeg-main/lab"
  python build_dataset_structure.py

  # 云端 (AutoDL):
  export MS_CRACKSEG_ROOT=/root/autodl-tmp/seg_pipeline/cloud_upload
  export MS_CRACKSEG_DATA=/root/autodl-tmp/seg_pipeline
  python build_dataset_structure.py

划分规则 (7:2:1, seed=42, v5.1: 308 有效场景, 19 个空标注已剔除):
  train(214) + val(60) -> img/ + masknpy/      (训练池, 274 场景)
  test(34)             -> testimg/ + testmasknpy/
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MASKS_DIR, SPLIT_PATH, CONFIGS_DIR, DATA_ROOT  # noqa: E402
from build_input_tensor import load_6channel  # noqa: E402

OUT = Path(os.environ.get("MS_DATASET_OUT", str(DATA_ROOT / "dataset_3class")))

CLASS_NAMES = ["background", "crack", "biological_attachment", "flaking"]


def export_scene(stem: str, category: str, img_dir: Path, mask_dir: Path):
    """导出单场景: 6ch float32 img npy + uint8 mask npy. 返回 (ok, err)."""
    out_img = img_dir / f"{stem}.npy"
    out_mask = mask_dir / f"{stem}.npy"
    if out_img.exists() and out_mask.exists():
        return True, "skip"
    try:
        tensor = load_6channel(stem, category)          # (6, H, W) float32
    except Exception as e:
        return False, f"img fail: {e}"
    mask_path = MASKS_DIR / category / f"{stem}_mask.png"
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, f"mask missing: {mask_path}"
    np.save(out_img, tensor.astype(np.float32))
    np.save(out_mask, mask.astype(np.uint8))
    return True, "ok"


def main():
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    img_dir = OUT / "img"; img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = OUT / "masknpy"; mask_dir.mkdir(parents=True, exist_ok=True)
    timg_dir = OUT / "testimg"; timg_dir.mkdir(parents=True, exist_ok=True)
    tmask_dir = OUT / "testmasknpy"; tmask_dir.mkdir(parents=True, exist_ok=True)

    pool = [(s["stem"], s["category"]) for s in split["train"] + split["val"]]
    test = [(s["stem"], s["category"]) for s in split["test"]]

    # 清理不在本次划分中的过期 npy (如空标注样本被排除后残留的文件)
    valid_pool = {stem for stem, _ in pool}
    valid_test = {stem for stem, _ in test}
    stale = []
    for d, valid in [(img_dir, valid_pool), (mask_dir, valid_pool),
                     (timg_dir, valid_test), (tmask_dir, valid_test)]:
        for p in d.glob("*.npy"):
            if p.stem not in valid:
                stale.append(p)
    for p in stale:
        try:
            p.unlink()
            print(f"  removed stale: {p}")
        except OSError as e:
            print(f"  [警告] 删除过期文件失败: {p} -> {e}")
    if stale:
        print(f"  removed {len(stale)} stale npy files")

    print(f"Output: {OUT}")
    print(f"Pool (train+val): {len(pool)} scenes -> img/ + masknpy/")
    print(f"Test: {len(test)} scenes -> testimg/ + testmasknpy/")

    fails = []
    for i, (stem, cat) in enumerate(pool, 1):
        ok, err = export_scene(stem, cat, img_dir, mask_dir)
        if not ok:
            fails.append((stem, cat, err)); print(f"  [{i}/{len(pool)}] FAIL {stem}: {err}")
        elif i % 25 == 0 or i == len(pool):
            print(f"  [{i}/{len(pool)}] pool done")

    for i, (stem, cat) in enumerate(test, 1):
        ok, err = export_scene(stem, cat, timg_dir, tmask_dir)
        if not ok:
            fails.append((stem, cat, err)); print(f"  [{i}/{len(test)}] FAIL {stem}: {err}")
        elif i % 10 == 0 or i == len(test):
            print(f"  [{i}/{len(test)}] test done")

    # 复制配置与说明
    shutil.copy(SPLIT_PATH, OUT / "split.json")
    for name in ["norm_stats.json", "norm_stats_pan.json"]:
        src = CONFIGS_DIR / name
        if src.exists():
            shutil.copy(src, OUT / name)

    readme = f"""# dataset_3class 数据集 (v2 4 类, {len(pool)}+{len(test)} 场景)

参考 stacked_5band_indices_842 结构. 更新时间见文件夹时间戳.

## 目录
- img/       训练池影像, {{stem}}.npy, shape=(6, 1088, 1456), float32
             通道顺序: [b1, b2, b3, b4, b5, pan]
             b1-b5 为反射率(约 0~1, 含少量负值与高光), pan 为暗校正 DN (uint16 数值域)
- masknpy/   训练池 mask, {{stem}}.npy, shape=(1088, 1456), uint8
- testimg/   测试集影像 (格式同 img/)
- testmasknpy/ 测试集 mask (格式同 masknpy/)

## 类别 (4 类, hollow 已移除)
0=background, 1=crack, 2=biological_attachment, 3=flaking
mask 绘制优先级: bio -> flaking -> crack (crack 最后画, 不被覆盖)

## 划分
train({len(split['train'])}) + val({len(split['val'])}) 场景在 img/masknpy (训练池);
test({len(split['test'])}) 场景在 testimg/testmasknpy.
train/val 边界以 split.json 的 train/val 键为准 (与参考工程在 img 池内
按 keys 划分 train/val 的做法一致). seed=42, 场景级分层 7:2:1.

## 归一化
z-score 参数在 norm_stats.json (6 通道, 仅 train split 计算) 与
norm_stats_pan.json (pan 通道), 加载后请按通道归一化再送入模型.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print()
    print(f"Done. Output: {OUT}")
    print(f"  img:        {len(list(img_dir.glob('*.npy')))} files")
    print(f"  masknpy:    {len(list(mask_dir.glob('*.npy')))} files")
    print(f"  testimg:    {len(list(timg_dir.glob('*.npy')))} files")
    print(f"  testmasknpy:{len(list(tmask_dir.glob('*.npy')))} files")
    print(f"  failures:   {len(fails)}")
    for f_ in fails:
        print("   ", f_)
    # 磁盘占用
    total = sum(f.stat().st_size for f in OUT.rglob("*.npy"))
    print(f"  total size: {total/1e9:.1f} GB")


if __name__ == "__main__":
    main()
