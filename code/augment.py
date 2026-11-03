# -*- coding: utf-8 -*-
"""
数据增强模块（云端版）
======================
几何增强(同步 image+mask) + 光谱增强(仅 image) + 类别平衡采样权重。

审计修复要点:
  - P1#6: 去掉 90° 随机旋转(违反建筑立面竖直先验), 改为 ±15° 仿射小角度旋转;
          vflip 概率降至 0.1 (墙倒置在现实中不存在, 仅作弱增强保留);
          hflip 保留(墙面左右翻转不影响语义).
  - P1#8: 光谱增强(gamma)需要反射率域 [0,1] 数据, 但 augmentor 接收的是 z-score
          后的数据. 正确做法是在 build_input_tensor 归一化之前做光谱增强.
          本模块新增 normalize=False 参数支持两种模式:
            * normalize=True (默认, 向后兼容): 仅做 noise + jitter (对 z-score 数据安全),
              gamma 被自动跳过并打印一次 warning.
            * normalize=False: 在原始反射率上做 gamma/noise/jitter, 调用方负责后续 z-score.
  - P2#11: 类别像素频率从 config.CLASS_PIXEL_COUNTS 加载, 不再硬编码.
"""
import numpy as np
import cv2
from pathlib import Path

from config import MASKS_DIR, CLASS_NAMES, CLASS_PIXEL_COUNTS


class SegmentationAugmentor:
    """6通道 image (6,H,W) + mask (H,W) 同步增强。

    Args:
        crop_size: 随机裁剪尺寸
        post_normalized: True 表示输入 image 已 z-score 归一化(默认);
                         False 表示输入是原始反射率 [0,1], 调用方后续自行归一化.
    """

    def __init__(self, crop_size=512,
                 prob_hflip=0.5, prob_vflip=0.1, prob_rotate90=0.0,
                 prob_small_rotate=0.5, rotate_range=15,
                 prob_scale=0.3, scale_range=(0.8, 1.2),
                 prob_gamma=0.5, gamma_range=(0.8, 1.2),
                 prob_noise=0.3, noise_std=0.02,
                 prob_jitter=0.3, jitter_std=0.03,
                 post_normalized=False):
        self.crop_size = crop_size
        self.prob_hflip = prob_hflip
        self.prob_vflip = prob_vflip
        self.prob_rotate90 = prob_rotate90          # 默认关闭: 违反立面先验
        self.prob_small_rotate = prob_small_rotate   # ±15° 仿射旋转
        self.rotate_range = rotate_range
        self.prob_scale = prob_scale
        self.scale_range = scale_range
        self.prob_gamma = prob_gamma
        self.gamma_range = gamma_range
        self.prob_noise = prob_noise
        self.noise_std = noise_std
        self.prob_jitter = prob_jitter
        self.jitter_std = jitter_std
        # N3 修复: 默认 post_normalized=False (原始反射率域).
        # train.py 的 SegDataset 已改为先 augment 原始张量再 normalize,
        # 因此 gamma 变换在 [0,1] 反射率上物理正确.
        # post_normalized=True 模式仅用于 val/test (无 gamma, 仅 noise/jitter).
        self.post_normalized = post_normalized

    def __call__(self, image, mask):
        image, mask = self._random_crop(image, mask)
        if np.random.rand() < self.prob_hflip:
            image = image[:, :, ::-1].copy(); mask = mask[:, ::-1].copy()
        if np.random.rand() < self.prob_vflip:
            image = image[:, ::-1, :].copy(); mask = mask[::-1, :].copy()
        if np.random.rand() < self.prob_rotate90:
            # 默认 prob=0 不会进入; 保留参数供消融实验
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        if np.random.rand() < self.prob_small_rotate:
            image, mask = self._small_rotate(image, mask)
        if np.random.rand() < self.prob_scale:
            image, mask = self._random_scale(image, mask)
        # ── 光谱增强 ──
        if not self.post_normalized:
            # 原始反射率域: gamma 变换物理正确
            # N3b 修复: gamma 只对 b1-b5 (channel 0-4, 反射率 [0,1]),
            #          pan (channel 5, uint16 0-65535) 跳过 gamma — 原代码
            #          np.clip(image, 0, 1) 会把 pan 全部 clip 成 0.
            if np.random.rand() < self.prob_gamma:
                gamma = np.random.uniform(*self.gamma_range, size=(5, 1, 1)).astype(np.float32)
                image[:5] = np.clip(image[:5], 0, 1) ** gamma
            if np.random.rand() < self.prob_noise:
                image = image + np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            if np.random.rand() < self.prob_jitter:
                offsets = np.random.normal(0, self.jitter_std, size=(6, 1, 1)).astype(np.float32)
                image = image + offsets
        else:
            # z-score 域: gamma 在均值 0 附近行为异常, 跳过; noise/jitter 安全
            # (此模式用于 val/test 无增强场景, train.py 的 SegDataset 默认用 False)
            if np.random.rand() < self.prob_noise:
                image = image + np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            if np.random.rand() < self.prob_jitter:
                offsets = np.random.normal(0, self.jitter_std, size=(6, 1, 1)).astype(np.float32)
                image = image + offsets
        return image.astype(np.float32), mask.astype(np.uint8)

    def _small_rotate(self, image, mask):
        """±rotate_range 度仿射旋转 (image 用 INTER_LINEAR, mask 用 INTER_NEAREST)."""
        angle = np.random.uniform(-self.rotate_range, self.rotate_range)
        _, H, W = image.shape
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img_r = np.stack([
            cv2.warpAffine(image[c], M, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            for c in range(6)], axis=0)
        mask_r = cv2.warpAffine(mask, M, (W, H),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return img_r, mask_r

    def _random_crop(self, image, mask):
        _, H, W = image.shape
        cs = self.crop_size
        if H < cs or W < cs:
            pad_h, pad_w = max(0, cs - H), max(0, cs - W)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
            H, W = image.shape[1], image.shape[2]
        y = np.random.randint(0, H - cs + 1)
        x = np.random.randint(0, W - cs + 1)
        return image[:, y:y + cs, x:x + cs], mask[y:y + cs, x:x + cs]

    def _random_scale(self, image, mask):
        _, H, W = image.shape
        scale = np.random.uniform(*self.scale_range)
        nh, nw = int(H * scale), int(W * scale)
        img_s = np.stack([cv2.resize(image[c], (nw, nh), interpolation=cv2.INTER_LINEAR)
                          for c in range(6)], axis=0)
        mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return self._center_crop_or_pad(img_s, mask_s, H, W)

    def _center_crop_or_pad(self, image, mask, th, tw):
        _, H, W = image.shape
        y0, x0 = max(0, (H - th) // 2), max(0, (W - tw) // 2)
        image = image[:, y0:y0 + th, x0:x0 + tw]
        mask = mask[y0:y0 + th, x0:x0 + tw]
        _, H, W = image.shape
        ph, pw = th - H, tw - W
        if ph > 0 or pw > 0:
            image = np.pad(image, ((0, 0), (0, max(0, ph)), (0, max(0, pw))), mode="reflect")
            mask = np.pad(mask, ((0, max(0, ph)), (0, max(0, pw))), mode="reflect")
        return image, mask


def compute_sample_weights(scene_list, mask_dir=None):
    """计算每场景的采样权重（稀有类场景权重高）。

    类别频率从 config.CLASS_PIXEL_COUNTS 读取 (审计 P2#11: 不再硬编码).
    """
    if mask_dir is None:
        mask_dir = MASKS_DIR
    mask_dir = Path(mask_dir)
    class_freq = CLASS_PIXEL_COUNTS
    total = sum(class_freq.values())
    cw = {c: total / (f * len(class_freq)) for c, f in class_freq.items()}
    max_w = max(cw.values())
    cw = {c: w / max_w for c, w in cw.items()}
    weights = []
    for s in scene_list:
        mask_path = mask_dir / s["category"] / f"{s['stem']}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            weights.append(1.0); continue
        n = mask.size
        sw = sum(cw[cls] * (int(np.sum(mask == i)) / n)
                 for i, cls in enumerate(CLASS_NAMES))
        weights.append(1.0 + sw)
    return weights
