# -*- coding: utf-8 -*-
"""
统一路径配置 — 通过环境变量适配本地 / AutoDL 云端环境
=========================================================
本地 (Windows):
  无需设置环境变量，默认路径相对于本文件所在目录

AutoDL (Linux):
  export MS_CRACKSEG_ROOT=/root/MS-CrackSeg          # 代码+mask+configs
  export MS_CRACKSEG_DATA=/root/autodl-tmp/data       # calibrated 数据（大数据盘）

审计修复要点:
  - P0#3: 新增 IGNORE_INDEX=255, 用于标注非墙面区域(天空/窗户/门洞)的忽略标签.
          CombinedLoss / MetricTracker / BoundaryLoss 均已支持跳过该标签.
  - P2#11: 新增 CLASS_PIXEL_COUNTS 字典, 统一类别像素频率数据源(augment/train 共用).
  - P2#12: 新增 CLASS_COLORS_RGB, 与原有 CLASS_COLORS_BGR 并存, 消除颜色表不一致.
  - P2#16: GSD 默认值移入 config, inference.py 通过参数覆盖.
"""
import os
from pathlib import Path

# ── 根目录 ──────────────────────────────────────────────────
# config.py 位于 code/ 目录，ROOT 指向 cloud_upload/
_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MS_CRACKSEG_ROOT", str(_ROOT)))

# 代码目录（config.py 所在目录）
CODE_DIR = Path(__file__).resolve().parent

# ── 数据目录 ────────────────────────────────────────────────
# calibrated 数据（大，放数据盘）
_DATA_DEFAULT = str(ROOT / "data") if (ROOT / "data" / "calibrated").exists() else "/root/autodl-tmp/data"
DATA_ROOT = Path(os.environ.get("MS_CRACKSEG_DATA", _DATA_DEFAULT))
CALIBRATED_DIR = DATA_ROOT / "calibrated"

# mask 数据（小，和代码一起）
MASKS_DIR = ROOT / "data" / "masks"

# 配置文件
CONFIGS_DIR = ROOT / "configs"
NORM_STATS_PATH = CONFIGS_DIR / "norm_stats.json"
SPLIT_PATH = CONFIGS_DIR / "split.json"
MEDIAN_H_PATH = CONFIGS_DIR / "median_fallback_H.json"
CLASS_FREQ_PATH = CONFIGS_DIR / "class_pixel_counts.json"

# 输出目录
OUTPUT_DIR = ROOT / "outputs"

# ── 模型常量 ────────────────────────────────────────────────
CLASS_NAMES = ["background", "crack", "biological_attachment", "flaking", "hollow"]
NUM_CLASSES = 5
H, W = 1088, 1456
BAND_NAMES = ["b1", "b2", "b3", "b4", "b5", "pan"]

# ── P0#3: ignore_index ─────────────────────────────────────
# 非墙面区域(天空/窗户/门洞/阳台/地面/植被)标注为 255, 训练/评估时跳过.
# CombinedLoss / MetricTracker 已支持. 标注规范: 0-4=墙面类, 255=非墙面.
IGNORE_INDEX = 255

# ── P2#16: 默认地面采样距离 ────────────────────────────────
DEFAULT_GSD_MM = 5.0

# ── P2#11: 类别像素频率 (单一数据源) ───────────────────────
# 优先从 configs/class_pixel_counts.json 加载; 若不存在则用内置默认值.
def _load_class_pixel_counts():
    try:
        import json
        if CLASS_FREQ_PATH.exists():
            with open(CLASS_FREQ_PATH) as f:
                data = json.load(f)
            return {k: int(v) for k, v in data.items()}
    except Exception:
        pass
    # 内置默认值 (与历史代码一致, 应定期从 configs/class_pixel_counts.json 更新)
    return {
        "background": 501586446,
        "crack": 9719840,
        "biological_attachment": 10129403,
        "flaking": 5624990,
        "hollow": 2038073,
    }

CLASS_PIXEL_COUNTS = _load_class_pixel_counts()

# ── P2#12: 颜色表 (RGB + BGR 双份, 消除不一致) ─────────────
# RGB 版本 (用于 matplotlib / PIL)
CLASS_COLORS_RGB = {
    0: (0, 0, 0),         # background - black
    1: (231, 76, 60),     # crack - red
    2: (52, 152, 219),    # biological_attachment - blue
    3: (46, 204, 113),    # flaking - green
    4: (243, 156, 18),    # hollow - orange
}
# BGR 版本 (用于 OpenCV cv2.imwrite)
CLASS_COLORS_BGR = {
    0: (0, 0, 0),
    1: (60, 76, 231),     # = RGB(231,76,60) reversed
    2: (219, 152, 52),    # = RGB(52,152,219) reversed
    3: (113, 204, 46),    # = RGB(46,204,113) reversed
    4: (18, 156, 243),    # = RGB(243,156,18) reversed
}


def print_paths():
    """打印当前路径配置（用于调试）"""
    print(f"ROOT:          {ROOT}")
    print(f"CODE_DIR:      {CODE_DIR}")
    print(f"DATA_ROOT:     {DATA_ROOT}")
    print(f"CALIBRATED:    {CALIBRATED_DIR}")
    print(f"MASKS:         {MASKS_DIR}")
    print(f"CONFIGS:       {CONFIGS_DIR}")
    print(f"OUTPUT:        {OUTPUT_DIR}")
    print(f"IGNORE_INDEX:  {IGNORE_INDEX}")
    print(f"CLASS_PIXEL_COUNTS: {CLASS_PIXEL_COUNTS}")


if __name__ == "__main__":
    print_paths()
