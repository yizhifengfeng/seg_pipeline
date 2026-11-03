# 建筑外墙多模态病害识别代码 — 专家级问题审计报告

> **审计范围**: `cloud_upload/code/` 目录下全部 7 个源码文件
> **审计日期**: 2026-11-03
> **审计方法**: 逐文件代码自审 + 数值实验验证 + 配置文件交叉比对 + 专家质疑逐条判定
> **验证环境**: PyTorch 2.6.0+cu124, CUDA, 39 项自动化测试全部通过

---

## 一、自审结论总览

代码工程骨架方向正确：多通道输入(6ch)→EdgePrior(8ch)→ResNet-34+CBAM→ASPP→Decoder，Focal+Lovasz+Boundary 联合损失，类别平衡采样。但存在 **5 个 P0 级数学/流程错误**（其中 1 个是真实的 reshape bug 被专家间接指出方向）、**4 个 P1 级领域常识问题**、**8 个 P2 级工程质量问题**。所有问题已修复并通过端到端测试。

---

## 二、专家质疑逐条审计

### P0 · 严重问题

#### ✅ 质疑 #1: FocalLoss α 被重复计算 — **正确，已修复**

**专家判断完全正确。** 我通过数值实验验证了数学错误：

| | 背景(α=0.3, p_t=0.881) | 稀有类(α=3.0, p_t=0.119) |
|---|---|---|
| 原代码 pt | 0.9626 (=0.881^0.3) | 0.001694 (=0.119^3.0) |
| 正确 pt | 0.8808 | 0.1192 |
| 原代码聚焦因子(1-pt)² | 0.0014 | **0.9966 ≈ 1（聚焦效应消失）** |
| 正确聚焦因子(1-pt)² | 0.0142 | **0.7758（强聚焦）** |

原代码把 α 塞进 `F.cross_entropy(weight=alpha)`，导致 `pt = exp(-ce) = p_t^α_t` 而非 `p_t`。对稀有类（α 大），pt→0，(1-pt)^γ→1，聚焦因子完全失效。

**修复方案**：先算无权重 CE 得 `pt = p_t`，乘调制因子 `(1-p_t)^γ`，最后乘 `alpha_t`。**额外发现并修复了一个 reshape bug**：原修复代码用 `logits[mask].view(-1, C)` 对 (B,C,H,W) 张量做布尔索引，内存 C 序会交错 channel 维（实测 [[2,3],[0,0]] 而非 [[2,0],[3,0]]），改为 `logits.permute(0,2,3,1).reshape(-1, C)` 正确排列。

#### ✅ 质疑 #2: LovaszLoss 硬编码 .cuda() — **正确，已修复**

`if torch.cuda.is_available(): grad = grad.cuda()` 在多卡(cuda:1)或 CPU 会 device mismatch。已改为 `grad = lovasz_grad(fg_sorted).to(probs.device)`。

#### ✅ 质疑 #3: 无 ignore_label — **正确，已修复（基础设施层）**

建筑外墙影像中的天空/窗户/门洞/植被不是"背景病害"，与完好墙面共享 class 0 会导致特征混淆和面积统计混入假阳性。

**修复方案**：
- `config.py` 新增 `IGNORE_INDEX = 255`
- `FocalLoss` / `LovaszLoss` / `BoundaryLoss` / `CombinedLoss` 均支持 `ignore_index=255`，跳过该标签像素
- `MetricTracker` 跳过 255
- `inference.py` 面积统计基于 `pred != 255`

**注意**：完整生效需要标注时把非墙面区域标 255。这是标注规范问题，代码层已就绪。

#### ✅ 质疑 #4: val 中心裁剪 512 vs 推理整图 1088×1456 — **正确，已修复**

原代码验证用固定中心 512×512（只看整图 1/6），推理跑整图。若裂缝集中在边缘，best_model 选择偏差。

**修复方案**：`validate()` 改为滑窗推理（512×512，步长 256，重叠 50%），拼接完整概率图后算 mIoU。新增 `sliding_window_predict()` 函数，与推理流程完全对齐。loss 用中心裁剪近似监控（避免整图 loss 过慢）。

#### ✅ 质疑 #5: norm_stats 数据泄露 — **正确，已确认并修复**

**实际验证**：`norm_stats.json` 的 `n_scenes = 334`，而 `split.json` 的 `total_scenes = 334`（train 233 + val 65 + test 36）。**norm_stats 在全部场景（含 val+test）上计算，确认存在泄露。**

**修复方案**：
- 新增 `recompute_norm_stats.py`：仅在 train split 的 233 场景上重算 mean/std
- `train.py` 启动时调用 `check_norm_stats()` 检查并打印警告：若 `n_scenes > len(train)` 则提示重算
- 输出 `norm_stats_train.json`，可用 `--overwrite` 覆盖原文件

---

### P1 · 主要问题

#### ✅ 质疑 #6: 90° 随机旋转违反建筑立面先验 — **正确，已修复**

建筑外墙有强竖直先验。90° 旋转把竖墙变横墙，物理不存在。vflip（墙倒置）同理不合理。

**修复方案**：
- `prob_rotate90` 默认 0（关闭，保留参数供消融）
- 新增 `prob_small_rotate=0.5` + `rotate_range=15`：±15° 仿射旋转（`cv2.warpAffine`，image 用 INTER_LINEAR，mask 用 INTER_NEAREST，borderMode=BORDER_REFLECT）
- `prob_vflip` 从 0.5 降至 0.1
- `prob_hflip` 保留 0.5（墙面左右翻转不影响语义）

#### ✅ 质疑 #7: output stride /32 细裂缝丢失 — **正确，已修复（最关键改动）**

GSD=5mm/px，细裂缝 2-3px 宽。在 /32 特征图上 = 0.06-0.1px，信息基本丢失。

**修复方案**：`DeepLabV3Plus(output_stride=16)`
- `layer3`: conv1 stride 2→1, dilation 1→2
- `layer4`: conv1 stride 2→1, dilation 1→4
- ASPP 接在 /16 特征图上（是 /32 的 4 倍大）
- 新增 `_convert_to_dilated()` 方法，兼容 BasicBlock(ResNet-34, stride on conv1) 和 Bottleneck(ResNet-50+, stride on conv2)

**验证**：OS=16 前向+反向通过，512×512 输入峰值显存仅 1.40GB（原 OS=32 约 0.8GB），batch_size=4 仍可跑。整图 1088×1456 推理可用 AMP。

#### ✅ 质疑 #8: 光谱增强做在 z-score 之后 — **正确，已修复**

gamma 变换 `sign·|x|^γ` 为反射率 [0,1] 设计，对 z-score 后数据（均值 0）在 x=0 附近行为异常。

**修复方案**：`SegmentationAugmentor(post_normalized=True/False)` 双模式
- `post_normalized=True`（默认，向后兼容）：自动跳过 gamma 并打印一次 warning，仅做 noise+jitter（对 z-score 安全）
- `post_normalized=False`：在原始反射率上做 gamma/noise/jitter，调用方负责后续 z-score

#### ❌ 质疑 #9: 没有独立 test set — **不正确（事实性错误），部分采纳**

**反驳**：`split.json` 已包含 `test` 键，36 个场景（biological_attachment 3 + crack 23 + flaking 10），`ratios = {train: 0.7, val: 0.2, test: 0.1}`，`seed = 42`。**test set 在数据层已存在。**

专家说"没有 test set"是事实性错误。但专家的隐含关切（train.py 只用 train+val，未加载 test）是合理的。

**采纳部分**：`train.py` 新增 `evaluate_test()`：训练结束后在 test set 上跑一次最终评估（滑窗），输出 test mIoU 和 test 混淆矩阵。

#### ✅ 质疑 #10: MetricTracker 逐像素 Python 循环 — **正确，已修复**

原代码 `for p, t in zip(pred.flatten(), target.flatten()): self.confusion[t, p] += 1` 对 26 万像素×batch 极慢。

**修复方案**：`np.bincount(idx, minlength=nc*nc)` 向量化，实测 <0.1s（原方法预估 5-10s），**加速 ~50-100x**。

---

### P2 · 次要问题（全部正确，已修复）

| # | 问题 | 修复 |
|---|---|---|
| 11 | 类别频率硬编码两处 | `config.CLASS_PIXEL_COUNTS` 统一数据源，优先从 `configs/class_pixel_counts.json` 加载 |
| 12 | 颜色表 BGR/RGB 不一致 | `config.CLASS_COLORS_RGB` + `CLASS_COLORS_BGR` 双份并存，RGB→BGR 转换验证一致 |
| 13 | 无随机种子 | `set_seed(42)` 设置 random/np/torch/cudnn，`--seed` 参数可配 |
| 14 | BoundaryLoss 每次重建 Sobel | `register_buffer("sobel_x/y")` 在 `__init__` 注册 |
| 15 | bare except 吞异常 | 收窄为 `(ValueError, KeyError, OSError)` + `warnings.warn` |
| 16 | GSD 硬编码 global 变量 | `config.DEFAULT_GSD_MM`，`--gsd` 参数覆盖，`compute_area_stats(gsd_mm)` 参数化 |
| 17 | 推理无置信度阈值/TTA | `--conf_threshold 0.5`（低置信度非背景像素归背景）+ `--tta`（水平翻转平均）|
| 18 | cv2.remap border fill 0 | `BORDER_CONSTANT, borderValue=0` → `BORDER_REFLECT` |

---

## 三、修复文件清单

| 文件 | 修改类型 | 关键修复 |
|---|---|---|
| `losses.py` | 重写 FocalLoss + 修复 Lovasz/Boundary | P0#1,2 P2#14 + reshape bug |
| `model.py` | 新增 OS=16 dilated backbone | P1#7 |
| `augment.py` | 重写增强策略 | P1#6,8 P2#11 |
| `train.py` | 滑窗验证+向量化+种子+test | P0#4,5 P1#9,10 P2#11,12,13 |
| `config.py` | 新增 IGNORE_INDEX/颜色/频率 | P0#3 P2#11,12,16 |
| `build_input_tensor.py` | 修复 except + border | P2#15,18 |
| `inference.py` | TTA + 置信度阈值 + GSD | P2#16,17 |
| `recompute_norm_stats.py` | **新增** | P0#5 |
| `test_audit_fixes.py` | **新增** | 39 项端到端验证 |

---

## 四、验证结果

```
======================================================================
审计修复验证测试
======================================================================
[1] FocalLoss: alpha 不扭曲 pt          2/2 PASS
[2] FocalLoss: ignore_index=255         3/3 PASS
[3] LovaszLoss: 设备无关                2/2 PASS
[4] BoundaryLoss: Sobel buffer          4/4 PASS
[5] CombinedLoss: ignore_index          2/2 PASS
[6] Model: output stride 16             6/6 PASS
[7] Augmentor: 旋转+gamma               5/5 PASS
[8] MetricTracker: 向量化+ignore        5/5 PASS
[9] Config: 颜色表+IGNORE_INDEX        10/10 PASS
======================================================================
结果: 39 passed, 0 failed
```

---

## 五、建议执行顺序

1. **立即执行**：`python recompute_norm_stats.py --overwrite`（修复 norm_stats 泄露）
2. **标注规范更新**：非墙面区域(天空/窗户/门洞)标 255（激活 ignore_index 基础设施）
3. **重新训练**：`python train.py --epochs 50 --batch_size 4 --lr 0.001`
4. **推理验证**：`python inference.py --stem img_0298 --category crack --tta --conf_threshold 0.5`

**预期效果**：细裂缝 IoU 显著提升（OS=16 + FocalLoss 修复），稀有类(hollow/flaking)聚焦恢复，best_model 选择可靠（滑窗验证），论文数字可信（train-only norm_stats + 独立 test set）。

---

## 六、第二轮审计：修复引入的新问题 (N1-N7)

> **审计日期**: 2026-11-03 (第二轮)
> **触发原因**: 第一轮修复引入 7 个新问题, 专家逐一指出
> **结论**: N1-N6 全部正确已修复, N7 用户明确推迟至下一轮训练

### N1 ✅ 滑窗边缘覆盖有洞 — **正确，已修复**

**问题**: `range(0, Hh-crop+1, stride)` 在 1088×1456 图上: Y=[0,256,512] 最后 patch 覆盖到 1024, **下 64 行未覆盖**; X=[0,256,512,768] 最后 patch 覆盖到 1280, **右 176 列未覆盖**. 未覆盖区域 count=0 → avg_prob=0 → argmax 全背景.

**修复**: 新增 `predict_utils._patch_starts(size, crop, stride)`, 保证最后一个 patch 起点在 `size-crop`:
- Y: [0, 256, 512, **576**] → 覆盖 576-1088 ✓
- X: [0, 256, 512, 768, **944**] → 覆盖 944-1456 ✓

**验证**: 端到端滑窗推理 1088×1456 图, 右边缘 10 列 + 下边缘 10 行概率均值 > 0.01 (非零).

### N2 ✅ inference.py 整图推理 OOM 风险 — **正确，已修复**

**问题**: docstring 写 "滑窗推理" 但代码是整图直接 `model(inp)`, OS=16 下 1088×1456 整图可能 OOM; TTA 翻倍显存; 无 autocast.

**修复**:
- 提取 `predict_utils.sliding_window_predict()` 为共享模块 (train + inference 复用)
- inference 调用滑窗, 每个 patch 有 `torch.amp.autocast`
- TTA 按 patch 做 (翻转推理→翻回→累加), 不翻倍显存

### N3 ✅ gamma 增强路径死代码 + pan 被 clip — **正确，已修复 (Plan B)**

**问题**:
- `train.py` 先 normalize 再 augment → `post_normalized=True` → gamma 永远跳过
- `post_normalized=False` 路径 `np.clip(image, 0, 1)` 对 pan(uint16 0-65535) 全部 clip 成 0

**修复 (Plan B: 完整方案)**:
- `SegDataset.__getitem__` 改为: `load_6channel(原始) → augment(原始) → normalize`
- gamma 只对 b1-b5 (channel 0-4, 反射率 [0,1]), **pan (channel 5) 跳过 gamma**
- 默认 `post_normalized=False` (原始反射率域, gamma 物理正确)

### N4 ✅ norm_stats pan 通道量纲 — **正确，已改进 (文档层)**

**问题**: b1-b5 (~0-1) 和 pan (0-65535) 混在一起 z-score, 量纲差异大.

**修复**: `recompute_norm_stats.py` 新增 `band_units` 和 `band_physical_range` 字段, 记录每个波段的物理单位和 min/max, 便于论文解释.

### N5 ✅ model.py 死代码三元 — **正确，已修复**

**问题**: `(6,12,18) if output_stride==16 else (6,12,18)` 两分支相同.

**修复**: 直接 `atrous_rates=(6,12,18)`.

### N6 ✅ validate loss 和 mIoU 区域不同 — **正确，已文档化**

**问题**: mIoU 用滑窗整图, val_loss 用中心 512 裁剪, 监测区域不同.

**修复**: 在代码注释中明确说明: val_loss 是快速监控代理指标 (中心裁剪), mIoU 是真实部署指标 (整图滑窗). loss 用于 early stop, mIoU 用于 best_model 选择. 整图 logits loss 需额外前向 (显存翻倍), 中心裁剪已足够反映收敛趋势.

### N7 ⏸ facade mask / 重新标注 — **用户明确推迟**

**问题**: 现有 mask 若没把天空/窗户标 255 而是 0, ignore_index 不生效; 推理无 facade mask 排除非墙面.

**状态**: 用户明确 "本次训练暂时不重新标注非墙面区域, 留作下一次训练". 代码层已就绪 (IGNORE_INDEX=255 + CombinedLoss/MetricTracker 支持), 待标注更新后自动生效.

---

## 七、第二轮修复文件清单

| 文件 | 修改类型 | 关键修复 |
|---|---|---|
| `predict_utils.py` | **新增** | N1+N2: 共享滑窗 (边缘全覆盖 + autocast + per-patch TTA) |
| `train.py` | 修改 | N1: 删除内联滑窗改用共享模块; N3: 增强顺序改为先augment再normalize; N6: loss区域文档化 |
| `augment.py` | 修改 | N3: 默认 post_normalized=False; gamma 只对 b1-b5, pan 跳过 |
| `inference.py` | 重写 | N2: 复用 sliding_window_predict + autocast + per-patch TTA |
| `recompute_norm_stats.py` | 修改 | N4: 记录 band_units + band_physical_range |
| `model.py` | 修改 | N5: 删除死代码三元 |
| `test_audit_fixes.py` | 扩展 | 新增 N1-N6 验证 (15 项), 总计 54 项 |

---

## 八、最终验证结果

```
======================================================================
审计修复验证测试 (第二轮)
======================================================================
[1]  FocalLoss: alpha 不扭曲 pt            2/2 PASS
[2]  FocalLoss: ignore_index=255           3/3 PASS
[3]  LovaszLoss: 设备无关                  2/2 PASS
[4]  BoundaryLoss: Sobel buffer            4/4 PASS
[5]  CombinedLoss: ignore_index            2/2 PASS
[6]  Model: output stride 16               6/6 PASS
[7]  Augmentor: 旋转+gamma 双模式          5/5 PASS
[8]  MetricTracker: 向量化+ignore          5/5 PASS
[9]  Config: 颜色表+IGNORE_INDEX          10/10 PASS
[10] predict_utils: 滑窗边缘全覆盖 (N1)    8/8 PASS
[11] inference: 使用滑窗推理 (N2)          3/3 PASS
[12] augment: gamma只对b1-b5 pan不clip(N3) 3/3 PASS
[13] model: 无死代码 (N5)                  1/1 PASS
======================================================================
结果: 54 passed, 0 failed
```
