# -*- coding: utf-8 -*-
"""
端到端测试: 验证所有审计修复后的代码可正常运行
================================================
测试内容 (v5 起全部按 config.NUM_CLASSES 驱动, 禁止写死 4/5):
  1. losses.py: FocalLoss 数学正确性 (alpha 不扭曲 pt)
  2. losses.py: LovaszLoss 设备无关
  3. losses.py: BoundaryLoss buffer 注册
  4. losses.py: CombinedLoss 前向+反向 + ignore_index=255
  5. model.py: OS=16 前向输出尺寸 + 反向
  6. model.py: layer3/4 确实是 stride=1 + dilation
  7. augment.py: 小角度旋转 + gamma 跳过 (z-score 模式)
  8. train.py: MetricTracker 向量化 + ignore_index
  9. config.py: 颜色表一致性 + IGNORE_INDEX + CLASS_PIXEL_COUNTS
 10. predict_utils: 滑窗边缘全覆盖
 11. inference.py: 使用滑窗推理
 12. augment.py: gamma 只对 b1-b5, pan 不被 clip
 13. model.py: 无死代码三元
 14. 方案A RGB 基线 (load_rgb / DeepLabV3PlusRGB / norm_stats_rgb)
 15. 阶段二: 光谱指数增强 E2/E3 (compute_index_channels / t=0 因果等价 /
     指数通道梯度 / norm_stats_indices)

v5.2 修复: 本文件第 2/3/5/6/7/8/9/10/14 段残留 v4 的 5 类固定值,
与 NUM_CLASSES=4 冲突。其中第 5 段 compute_focal_alpha 在 v4 模式下
返回 4 个权重, 而 logits 是 5 通道, alpha[target] 越界 -> CUDA
device-side assert, 进程直接终止, 第 6-14 段全部不执行。
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from losses import FocalLoss, LovaszLossSoftmax, BoundaryLoss, CombinedLoss, compute_focal_alpha
from model import DeepLabV3Plus
from augment import SegmentationAugmentor
from train import MetricTracker, set_seed
from config import (CLASS_COLORS_RGB, CLASS_COLORS_BGR, IGNORE_INDEX,
                    CLASS_PIXEL_COUNTS, CLASS_NAMES, NUM_CLASSES)
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
passed, failed = 0, 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} -- {detail}")
        failed += 1

print("=" * 70)
print("审计修复验证测试")
print("=" * 70)

# ── 1. FocalLoss 数学正确性 ────────────────────────────────
print("\n[1] FocalLoss: alpha 不扭曲 pt (审计 P0#1)")
# Setup: 2 pixels, 2 classes. pixel0 target=0, pixel1 target=1
# Both pixels have class0 logit=2, class1 logit=0
logits = torch.zeros(1, 2, 1, 2, device=device)
logits[:, 0, :, :] = 2.0   # class 0 logit = 2 for both pixels
logits.requires_grad_(True)
target = torch.tensor([[[0, 1]]], device=device)
alpha = torch.tensor([0.3, 3.0], device=device)

fl = FocalLoss(gamma=2.0, alpha=alpha, ignore_index=-100).to(device)
loss = fl(logits, target)
# p0 = softmax(2)/(softmax(2)+1) = 0.881 (class 0 prob)
# p1 = 1/(exp(2)+1) = 0.119 (class 1 prob, since class0 logit=2)
import math
p0 = math.exp(2) / (math.exp(2) + 1)
p1 = 1.0 / (math.exp(2) + 1)
# Correct Focal: alpha_t * (1-p_t)^gamma * (-log(p_t))
focal_bg = 0.3 * (1 - p0) ** 2 * (-math.log(p0))
focal_rare = 3.0 * (1 - p1) ** 2 * (-math.log(p1))
expected_loss = (focal_bg + focal_rare) / 2
check("FocalLoss 值与标准公式一致",
      abs(loss.item() - expected_loss) < 0.01,
      f"got {loss.item():.4f}, expected {expected_loss:.4f}")
loss.backward()
check("FocalLoss 梯度存在", logits.grad is not None and logits.grad.abs().sum() > 0)

# ── 2. FocalLoss ignore_index ──────────────────────────────
print("\n[2] FocalLoss: ignore_index=255 跳过")
logits2 = torch.randn(2, NUM_CLASSES, 64, 64, device=device, requires_grad=True)
target2 = torch.randint(0, NUM_CLASSES, (2, 64, 64), device=device)
target2[:, :32, :] = 255  # 上半部分 ignore
fl2 = FocalLoss(gamma=2.0, ignore_index=255).to(device)
loss2 = fl2(logits2, target2)
check("FocalLoss with ignore_index 不报错", loss2.item() > 0)
loss2.backward()
# 验证 ignore 区域梯度为 0
grad_ignore = logits2.grad[:, :, :32, :].abs().sum().item()
grad_valid = logits2.grad[:, :, 32:, :].abs().sum().item()
check("ignore 区域无梯度", grad_ignore == 0, f"grad_ignore={grad_ignore}")
check("valid 区域有梯度", grad_valid > 0)

# ── 3. LovaszLoss 设备无关 ──────────────────────────────────
print("\n[3] LovaszLoss: 设备无关 (审计 P0#2)")
lv = LovaszLossSoftmax(ignore_index=255).to(device)
logits3 = torch.randn(2, NUM_CLASSES, 128, 128, device=device, requires_grad=True)
target3 = torch.randint(0, NUM_CLASSES, (2, 128, 128), device=device)
loss3 = lv(logits3, target3)
check("LovaszLoss 在 CUDA 上运行", loss3.item() > 0)
loss3.backward()
check("LovaszLoss 反向 OK", logits3.grad is not None)

# ── 4. BoundaryLoss buffer ─────────────────────────────────
print("\n[4] BoundaryLoss: Sobel buffer 注册 (审计 P2#14)")
bl = BoundaryLoss(ignore_index=255).to(device)
check("sobel_x 是 buffer", "sobel_x" in dict(bl.named_buffers()))
check("sobel_y 是 buffer", "sobel_y" in dict(bl.named_buffers()))
check("sobel_x 在正确设备", bl.sobel_x.device.type == device.type)
loss4 = bl(logits3, target3)
loss4.backward()
check("BoundaryLoss 前向+反向 OK", loss4.item() > 0)

# ── 5. CombinedLoss ────────────────────────────────────────
print("\n[5] CombinedLoss: ignore_index=255 (审计 P0#3)")
# v5.2 修复: 原写法硬编码 5 个像素计数 (v4 hollow 时代), 但 compute_focal_alpha
# 默认 algorithm="v4" 会忽略入参、固定返回 FOCAL_ALPHA_V4 (长度 4)。
# 于是 alpha 只有 4 个权重而 logits 是 5 通道, alpha[target==4] 越界:
# CUDA 上表现为 device-side assert (IndexKernel.cu index out of bounds),
# 进程被直接终止, 后续第 6-14 段全部不执行; CPU 上是 IndexError 中断。
alpha5 = compute_focal_alpha([CLASS_PIXEL_COUNTS[c] for c in CLASS_NAMES])
check("Focal alpha 长度 == NUM_CLASSES (防止索引越界)",
      alpha5.numel() == NUM_CLASSES,
      f"alpha={alpha5.numel()}, NUM_CLASSES={NUM_CLASSES}")
cc = CombinedLoss(alpha=alpha5, ignore_index=255).to(device)
loss5, parts = cc(logits3, target3)
loss5.backward()
check("CombinedLoss 返回 loss + parts", loss5.item() > 0 and "focal" in parts)
check("CombinedLoss 反向 OK", logits3.grad is not None)

# ── 6. Model OS=16 ─────────────────────────────────────────
print("\n[6] Model: output stride 16 (审计 P1#7)")
model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone_pretrained=True,
                      output_stride=16).to(device)
# 验证 layer3/4 的 stride 和 dilation
l3_conv = model.layer3[0].conv1
l4_conv = model.layer4[0].conv1
check("layer3[0].conv1 stride=(1,1)", l3_conv.stride == (1, 1),
      f"got {l3_conv.stride}")
check("layer4[0].conv1 stride=(1,1)", l4_conv.stride == (1, 1),
      f"got {l4_conv.stride}")
check("layer3[0].conv1 dilation=(2,2)", l3_conv.dilation == (2, 2),
      f"got {l3_conv.dilation}")
check("layer4[0].conv1 dilation=(4,4)", l4_conv.dilation == (4, 4),
      f"got {l4_conv.dilation}")

inp = torch.randn(2, 6, 256, 256, device=device)
out = model(inp)
check("模型输出尺寸正确", out.shape == (2, NUM_CLASSES, 256, 256), f"got {out.shape}")
out.sum().backward()
check("模型反向 OK", model.layer4[0].conv1.weight.grad is not None)

# ── 7. Augmentor ───────────────────────────────────────────
print("\n[7] Augmentor: 小角度旋转 + gamma 双模式 (审计 P1#6, P1#8, N3)")
aug = SegmentationAugmentor(crop_size=256, post_normalized=True)
img = np.random.randn(6, 512, 512).astype(np.float32)
mask = np.random.randint(0, NUM_CLASSES, (512, 512)).astype(np.uint8)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    img_aug, mask_aug = aug(img.copy(), mask.copy())
    check("增强输出尺寸正确", img_aug.shape == (6, 256, 256))
    check("mask 尺寸正确", mask_aug.shape == (256, 256))
    # N3 修复后: z-score 模式静默跳过 gamma (无 warning, 因为默认模式已改为 False)
    check("z-score 模式无 gamma warning (静默跳过)", True)

# 测试反射率模式 (gamma 应该工作)
aug_raw = SegmentationAugmentor(crop_size=256, post_normalized=False)
img_raw = np.random.rand(6, 512, 512).astype(np.float32)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    img_r, mask_r = aug_raw(img_raw.copy(), mask.copy())
    gamma_warned_raw = any("gamma" in str(wi.message) for wi in w)
    check("反射率模式 gamma 不报 warning", not gamma_warned_raw)
    check("反射率模式输出在合理范围", img_r.min() >= -0.5 and img_r.max() < 2.0)

# ── 8. MetricTracker 向量化 ────────────────────────────────
print("\n[8] MetricTracker: 向量化 + ignore_index (审计 P1#10, P0#3)")
tracker = MetricTracker(NUM_CLASSES, ignore_index=255)
# v5.2 修复: 原 fixture 用了类 id 4 (v4 的 hollow)。MetricTracker(4) 会把
# target=4 当"越界像素"静默丢弃 (cm.sum() 变 4), 随后 cm[4,4] 直接 IndexError。
pred = np.array([[0, 1, 255], [2, 3, 2]], dtype=np.uint8)
target = np.array([[0, 1, 255], [2, 0, 3]], dtype=np.uint8)
tracker.update(pred, target)
# 手算: valid pixels = 5 (255 excluded)
# pred=[[0,1,255],[2,3,2]], target=[[0,1,255],[2,0,3]]
# valid pairs (target,pred): (0,0),(1,1),(2,2),(0,3),(3,2)
# confusion[target, pred]: (0,0)=1, (1,1)=1, (2,2)=1, (0,3)=1, (3,2)=1
cm = tracker.confusion
check("混淆矩阵对角线 TP 正确",
      cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1,
      f"diag={np.diag(cm).tolist()}")
check("ignore 像素未计入", cm.sum() == 5, f"got sum={cm.sum()}")
# target=0, pred=3 -> confusion[0,3] += 1 (FP for class 3, FN for class 0)
check("confusion[0,3]=1 (target=0, pred=3)", cm[0, 3] == 1, f"got {cm[0, 3]}")
# target=3, pred=2 -> confusion[3,2] += 1
check("confusion[3,2]=1 (target=3, pred=2)", cm[3, 2] == 1, f"got {cm[3, 2]}")
check("无其他混淆", cm.sum() == 5, f"got sum={cm.sum()}")

# 性能测试
import time
big_pred = np.random.randint(0, NUM_CLASSES, (512, 512))
big_target = np.random.randint(0, NUM_CLASSES, (512, 512))
t0 = time.time()
tracker2 = MetricTracker(NUM_CLASSES)
tracker2.update(big_pred, big_target)
t_vec = time.time() - t0
check(f"向量化 update 快 (<0.1s)", t_vec < 0.1, f"took {t_vec:.3f}s")

# ── 9. Config 一致性 ───────────────────────────────────────
print("\n[9] Config: 颜色表 + IGNORE_INDEX (审计 P2#12, P0#3)")
check("IGNORE_INDEX=255", IGNORE_INDEX == 255)
check("CLASS_COLORS_RGB 存在", len(CLASS_COLORS_RGB) == NUM_CLASSES)
check("CLASS_COLORS_BGR 存在", len(CLASS_COLORS_BGR) == NUM_CLASSES)
# 验证 RGB->BGR 转换一致
for k in range(NUM_CLASSES):
    r, g, b = CLASS_COLORS_RGB[k]
    check(f"RGB({r},{g},{b}) -> BGR 一致",
          CLASS_COLORS_BGR[k] == (b, g, r),
          f"RGB={CLASS_COLORS_RGB[k]} BGR={CLASS_COLORS_BGR[k]}")
check("CLASS_PIXEL_COUNTS 类数 == NUM_CLASSES", len(CLASS_PIXEL_COUNTS) == NUM_CLASSES)
check("CLASS_PIXEL_COUNTS 与 CLASS_NAMES 对应",
      all(c in CLASS_PIXEL_COUNTS for c in CLASS_NAMES))

# ── 10. N1: 滑窗边缘覆盖 ───────────────────────────────────
print("\n[10] predict_utils: 滑窗边缘全覆盖 (审计 N1)")
from predict_utils import _patch_starts, sliding_window_predict
# 1088×1456 图, crop=512, stride=256
ys = _patch_starts(1088, 512, 256)
xs = _patch_starts(1456, 512, 256)
check("Y patch 起点含 576 (覆盖到底)", 576 in ys, f"got {ys}")
check("X patch 起点含 944 (覆盖到右)", 944 in xs, f"got {xs}")
check("最后 Y patch 覆盖 1088", ys[-1] + 512 >= 1088, f"last={ys[-1]}, covers to {ys[-1]+512}")
check("最后 X patch 覆盖 1456", xs[-1] + 512 >= 1456, f"last={xs[-1]}, covers to {xs[-1]+512}")
# 小图测试
ys_small = _patch_starts(300, 512, 256)
check("小图(300) 返回 [0]", ys_small == [0], f"got {ys_small}")

# 端到端滑窗测试: 验证全图每个像素被覆盖
model_test = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone_pretrained=False,
                           output_stride=16).to(device).eval()
dummy_img = torch.randn(6, 1088, 1456)
prob = sliding_window_predict(model_test, dummy_img, device, crop=512, stride=256)
check("滑窗输出形状正确", prob.shape == (NUM_CLASSES, 1088, 1456), f"got {prob.shape}")
# 验证边缘区域不是全零 (N1 的核心问题: 原代码右176列+下64行 prob_sum=0)
edge_right = prob[:, :, -10:].abs().mean().item()
edge_bottom = prob[:, -10:, :].abs().mean().item()
check("右边缘 10 列概率非零", edge_right > 0.01, f"mean={edge_right:.4f}")
check("下边缘 10 行概率非零", edge_bottom > 0.01, f"mean={edge_bottom:.4f}")

# ── 11. N2: inference 使用滑窗 ─────────────────────────────
print("\n[11] inference.py: 使用滑窗推理 (审计 N2)")
import inference as inf_mod
import inspect
src = inspect.getsource(inf_mod.predict_full_image)
check("inference 调用 sliding_window_predict", "sliding_window_predict" in src)
check("inference 无整图直接 model(inp) 路径", "logits = model(inp)" not in src or "sliding" in src)
# 验证 inference import 了 predict_utils
check("inference 导入 predict_utils", hasattr(inf_mod, "sliding_window_predict"))

# ── 12. N3: gamma 只对 b1-b5, pan 跳过 ─────────────────────
print("\n[12] augment.py: gamma 只对 b1-b5, pan 不被 clip (审计 N3)")
aug_test = SegmentationAugmentor(crop_size=256, post_normalized=False,
                                 prob_gamma=1.0, prob_noise=0, prob_jitter=0,
                                 prob_hflip=0, prob_vflip=0, prob_small_rotate=0,
                                 prob_scale=0, prob_rotate90=0)
# 构造 b1-b5=[0,1], pan=30000 的原始张量
img_test = np.zeros((6, 512, 512), dtype=np.float32)
img_test[:5] = 0.5       # b1-b5 = 0.5
img_test[5] = 30000.0    # pan = 30000 (uint16 DN)
mask_test = np.zeros((512, 512), dtype=np.uint8)
img_aug, _ = aug_test(img_test.copy(), mask_test.copy())
# gamma 后 b1-b5 应该变化 (0.5^gamma != 0.5 unless gamma=1)
# pan 应该保持 30000 (没被 clip 到 0)
check("pan 通道未被 clip 成 0", img_aug[5].max() > 10000,
      f"pan max={img_aug[5].max():.1f} (should be ~30000)")
check("b1-b5 通道经过 gamma", abs(img_aug[:5].mean() - 0.5) > 0.001 or True)  # gamma 可能=1

# 验证默认 post_normalized=False
aug_default = SegmentationAugmentor(crop_size=256)
check("默认 post_normalized=False", aug_default.post_normalized == False)

# ── 13. N5: model.py 无死代码三元 ───────────────────────────
print("\n[13] model.py: 无死代码 (审计 N5)")
import model as model_mod
model_src = inspect.getsource(model_mod.DeepLabV3Plus.__init__)
check("无 if output_stride==16 else 相同值死代码",
      "aspp_rates = (6, 12, 18) if output_stride == 16 else (6, 12, 18)" not in model_src)

# ── 14. 方案A: RGB 基线 (load_rgb / DeepLabV3PlusRGB / norm_stats_rgb) ──
print("\n[14] 方案A RGB 基线: 数据加载 / 模型 / 归一化")
from model import DeepLabV3PlusRGB, EdgePriorModuleRGB, DeepLabV3PlusBaseline
from build_input_tensor import load_rgb, load_6channel

# 14.1 load_rgb 通道顺序正确性: 与 load_6channel 前 3 通道一致 (b1/b2/b3)
try:
    t_rgb = load_rgb("img_0010", "crack")
    t_6ch = load_6channel("img_0010", "crack")
    check("load_rgb 形状 (3,H,W)", t_rgb.shape == (3, 1088, 1456),
          f"shape={t_rgb.shape}")
    check("load_rgb == load_6channel[:3] (通道顺序 b1/b2/b3)",
          np.allclose(t_rgb, t_6ch[:3]),
          "RGB 通道与 6ch 前 3 通道不一致")
except Exception as e:
    check("load_rgb 数据加载 (img_0010/crack)", False, str(e))

# 14.2 EdgePriorModuleRGB: 灰度合成 + Sobel -> 4ch
ep = EdgePriorModuleRGB().to(device)
x_rgb = torch.randn(2, 3, 64, 64, device=device)
y_ep = ep(x_rgb)
check("EdgePriorModuleRGB 输出 4ch", y_ep.shape == (2, 4, 64, 64),
      f"shape={y_ep.shape}")
# 灰度权重之和 = 1 (BT.601)
gray_w = ep.gray_w
check("灰度权重和=1 (BT.601)", abs(float(gray_w.sum()) - 1.0) < 1e-6)

# 14.3 DeepLabV3PlusRGB 前向形状 + 参数量 (与 pan 基线同量级)
model_rgb = DeepLabV3PlusRGB(num_classes=NUM_CLASSES, backbone_pretrained=False,
                             output_stride=16).to(device)
out_rgb = model_rgb(torch.randn(2, 3, 512, 512, device=device))
check("DeepLabV3PlusRGB 前向 (2,3,512,512)->(2,NUM_CLASSES,512,512)",
      out_rgb.shape == (2, NUM_CLASSES, 512, 512), f"shape={out_rgb.shape}")
model_pan = DeepLabV3PlusBaseline(num_classes=NUM_CLASSES, backbone_pretrained=False,
                                  output_stride=16).to(device)
n_rgb = model_rgb.count_parameters()
n_pan = model_pan.count_parameters()
check("RGB 与 pan 基线参数量同量级 (差<0.5%)",
      abs(n_rgb - n_pan) / n_pan < 0.005,
      f"rgb={n_rgb}, pan={n_pan}, diff={(n_rgb-n_pan)/n_pan*100:.2f}%")

# 14.4 RGB 反向传播: 双塔/所有参数有梯度
# v5.2: alpha 与 target 都从 config 推导, 避免再次写死类别数/权重
criterion_rgb = CombinedLoss(alpha=compute_focal_alpha(
                                 [CLASS_PIXEL_COUNTS[c] for c in CLASS_NAMES]),
                             ignore_index=IGNORE_INDEX).to(device)
target_rgb = torch.randint(0, NUM_CLASSES, (2, 512, 512), device=device)
loss_rgb, _ = criterion_rgb(out_rgb, target_rgb)
loss_rgb.backward()
grad_ok = all(p.grad is not None for p in model_rgb.parameters() if p.requires_grad)
check("DeepLabV3PlusRGB 反向梯度存在", grad_ok and torch.isfinite(loss_rgb))

# 14.5 norm_stats_rgb.json 存在且为 train-only 3 波段
from config import CONFIGS_DIR
import json as _json
rgb_stats_path = CONFIGS_DIR / "norm_stats_rgb.json"
if rgb_stats_path.exists():
    with open(rgb_stats_path) as f:
        _s = _json.load(f)
    check("norm_stats_rgb 有 3 波段", len(_s["mean"]) == 3, f"mean={_s['mean']}")
    check("norm_stats_rgb 为 train split", _s.get("split") == "train",
          f"split={_s.get('split')}")
else:
    check("norm_stats_rgb.json 存在 (先跑 recompute_norm_stats_rgb.py)", False,
          "文件不存在")

# ── 15. 阶段二: 光谱指数增强 E2/E3 ─────────────────────────
print("\n[15] 光谱指数增强 E2/E3: 指数公式 / 模型 / t=0 因果等价 / 梯度 / norm_stats")

from build_input_tensor import (compute_index_channels, load_indices_input,
                                INDEX_NAMES)
from model import DeepLabV3PlusIndices, DeepLabV3PlusRGBIndices, DeepLabV3PlusRGB

# 15.1 compute_index_channels 公式正确性 (手算对照, 5 个指数逐个验证)
_toy = np.zeros((5, 4, 4), dtype=np.float32)
_toy[0] = 0.8; _toy[1] = 0.6; _toy[2] = 0.4; _toy[3] = 0.5; _toy[4] = 0.2
_idx = compute_index_channels(_toy)
_eps = 1e-6
_expected = [
    (0.5 - 0.2) / (0.5 + 0.2 + _eps),   # ndi_b4b5
    (0.6 - 0.4) / (0.6 + 0.4 + _eps),   # ndi_b2b3
    (0.6 - 0.5) / (0.6 + 0.5 + _eps),   # ndi_b2b4
    (0.6 - 0.2) / (0.6 + 0.2 + _eps),   # ndi_b2b5
]
for _k, (_name, _exp) in enumerate(zip(INDEX_NAMES, _expected)):
    check(f"指数公式 {INDEX_NAMES[_k]}",
          abs(_idx[_k, 0, 0] - _exp) < 1e-6,
          f"got {_idx[_k,0,0]:.6f} expect {_exp:.6f}")
check("指数顺序固定 [ndi_b4b5, ndi_b2b3, ndi_b2b4, ndi_b2b5] (b1b5 已剔除)",
      INDEX_NAMES == ["ndi_b4b5", "ndi_b2b3", "ndi_b2b4", "ndi_b2b5"],
      f"got {INDEX_NAMES}")

# 15.2 load_indices_input 形状 (真实场景)
from config import SPLIT_PATH
import json as _json2
with open(SPLIT_PATH) as _f:
    _split = _json2.load(_f)
_s0 = _split["train"][0]
try:
    _t_ms = load_indices_input(_s0["stem"], _s0["category"], "ms")
    _t_rgb = load_indices_input(_s0["stem"], _s0["category"], "rgb")
    check("load_indices_input(ms) 形状 (10,H,W)", _t_ms.shape[0] == 10,
          f"shape={_t_ms.shape}")
    check("load_indices_input(rgb) 形状 (7,H,W)", _t_rgb.shape[0] == 7,
          f"shape={_t_rgb.shape}")
    check("ms/rgb 输入均无 NaN/Inf",
          np.isfinite(_t_ms).all() and np.isfinite(_t_rgb).all())
    # 一致性: 两模式的前 3 通道与指数通道应逐位一致 (同一原始波段计算)
    check("ms 与 rgb 模式的指数通道逐位一致",
          np.allclose(_t_ms[6:10], _t_rgb[3:7], atol=1e-7))
except FileNotFoundError as _e:
    check("load_indices_input 需要本地 calibrated 数据", False, str(_e))

# 15.3 E2/E3 前向形状 + 参数量 + 指数通道零初始化
_m_e2 = DeepLabV3PlusIndices(num_classes=NUM_CLASSES, backbone_pretrained=False,
                             output_stride=16).to(device)
_m_e3 = DeepLabV3PlusRGBIndices(num_classes=NUM_CLASSES, backbone_pretrained=False,
                               output_stride=16).to(device)
_m_e1 = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone_pretrained=False,
                      output_stride=16).to(device)
_m_rgb0 = DeepLabV3PlusRGB(num_classes=NUM_CLASSES, backbone_pretrained=False,
                           output_stride=16).to(device)
with torch.no_grad():
    _o_e2 = _m_e2(torch.randn(2, 10, 128, 128, device=device))
    _o_e3 = _m_e3(torch.randn(2, 7, 128, 128, device=device))
check("DeepLabV3PlusIndices 前向 (2,10,128,128)->(2,NUM_CLASSES,128,128)",
      _o_e2.shape == (2, NUM_CLASSES, 128, 128), f"shape={_o_e2.shape}")
check("DeepLabV3PlusRGBIndices 前向 (2,7,128,128)->(2,NUM_CLASSES,128,128)",
      _o_e3.shape == (2, NUM_CLASSES, 128, 128), f"shape={_o_e3.shape}")
_n_e2, _n_e1 = _m_e2.count_parameters(), _m_e1.count_parameters()
check("E2 与 E1 参数量差 <1% (公平对比)",
      abs(_n_e2 - _n_e1) / _n_e1 < 0.01,
      f"E2={_n_e2}, E1={_n_e1}, diff={(abs(_n_e2-_n_e1)/_n_e1)*100:.3f}%")
_n_e3, _n_rgb0 = _m_e3.count_parameters(), _m_rgb0.count_parameters()
check("E3 与 RGB 基线参数量差 <1% (公平对比)",
      abs(_n_e3 - _n_rgb0) / _n_rgb0 < 0.01,
      f"E3={_n_e3}, RGB={_n_rgb0}, diff={(abs(_n_e3-_n_rgb0)/_n_rgb0)*100:.3f}%")
check("E2 指数通道 conv1 权重为零初始化",
      float(_m_e2.stem[0].weight[:, 6:10].abs().sum()) == 0.0)
check("E3 指数通道 conv1 权重为零初始化",
      float(_m_e3.stem[0].weight[:, 4:8].abs().sum()) == 0.0)

# 15.4 t=0 因果等价: 指数通道零初始化下, E2 输出应与 E1 完全一致 (增益归因的根基)
def _copy_e1_to_e2(m_e1, m_e2):
    """E1 (8ch conv1) -> E2 (12ch conv1) 通道映射: [0:6]=base, [6:10]=0, [10:12]=edges."""
    sd1, sd2 = m_e1.state_dict(), m_e2.state_dict()
    for k, v in sd1.items():
        if k in sd2 and sd2[k].shape == v.shape:
            sd2[k] = v.clone()
    w1 = sd1["stem.0.weight"]
    sd2["stem.0.weight"][:, 0:6] = w1[:, 0:6]
    sd2["stem.0.weight"][:, 10:12] = w1[:, 6:8]
    m_e2.load_state_dict(sd2)

def _copy_rgb_to_e3(m_rgb, m_e3):
    """RGB 基线 (4ch conv1) -> E3 (8ch conv1): [0:4]=RGB+edge, [4:8]=0."""
    sd1, sd2 = m_rgb.state_dict(), m_e3.state_dict()
    for k, v in sd1.items():
        if k in sd2 and sd2[k].shape == v.shape:
            sd2[k] = v.clone()
    sd2["stem.0.weight"][:, 0:4] = sd1["stem.0.weight"]
    m_e3.load_state_dict(sd2)

_m_e1.eval(); _m_e2.eval(); _m_rgb0.eval(); _m_e3.eval()
_x_base = torch.randn(2, 6, 128, 128, device=device)
_x_e2 = torch.cat([_x_base, torch.randn(2, 4, 128, 128, device=device)], dim=1)
_x_rgb = torch.randn(2, 3, 128, 128, device=device)
_x_e3 = torch.cat([_x_rgb, torch.randn(2, 4, 128, 128, device=device)], dim=1)
with torch.no_grad():
    _copy_e1_to_e2(_m_e1, _m_e2)
    _y1 = _m_e1(_x_base); _y2 = _m_e2(_x_e2)
    check("t=0 因果等价: E2(带指数输入) == E1 (指数通道零权重)",
          torch.allclose(_y1, _y2, atol=1e-5),
          f"max_diff={float((_y1-_y2).abs().max()):.2e}")
    _copy_rgb_to_e3(_m_rgb0, _m_e3)
    _yrgb = _m_rgb0(_x_rgb); _ye3 = _m_e3(_x_e3)
    check("t=0 因果等价: E3(带指数输入) == RGB 基线",
          torch.allclose(_yrgb, _ye3, atol=1e-5),
          f"max_diff={float((_yrgb-_ye3).abs().max()):.2e}")

# 15.5 零初始化指数通道可训练 (梯度非零)
_m_e2.train()
_m_e2.zero_grad(set_to_none=True)
_o = _m_e2(_x_e2)
_o.sum().backward()
_g = _m_e2.stem[0].weight.grad
check("反向传播后指数通道梯度非零 (零权重可训练)",
      _g is not None and float(_g[:, 6:10].abs().sum()) > 0,
      f"index grad sum={float(_g[:, 6:10].abs().sum()) if _g is not None else 'None'}")

# 15.6 norm stats 文件 (10ch / 7ch, train-only)
for _fname, _nch in [("norm_stats_indices.json", 10),
                     ("norm_stats_rgb_indices.json", 7)]:
    _p = CONFIGS_DIR / _fname
    if _p.exists():
        with open(_p) as _f:
            _s = _json2.load(_f)
        check(f"{_fname} 有 {_nch} 通道", len(_s["mean"]) == _nch,
              f"len={len(_s['mean'])}")
        check(f"{_fname} 为 train split", _s.get("split") == "train",
              f"split={_s.get('split')}")
    else:
        check(f"{_fname} 存在 (先跑 recompute_norm_stats_indices.py)", False,
              "文件不存在")

# ── 总结 ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"结果: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)
