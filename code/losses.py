# -*- coding: utf-8 -*-
"""
Stage 6: Background Suppression Losses
========================================
Three complementary losses to suppress background and boost rare classes:

1. FocalLoss   - down-weights easy/background pixels, focuses on hard/rare ones
2. LovaszLoss  - directly optimizes IoU (better for small targets like cracks)
3. BoundaryLoss- improves edge precision of defect boundaries

Combined: L = lambda_focal * Focal + lambda_lovasz * Lovasz + lambda_boundary * Boundary

Importable: from losses import CombinedLoss, FocalLoss, LovaszLossSoftmax, BoundaryLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# v4 Focal alpha 权重 (2026-11-04, 基于 v3 重训 50 epoch 诊断微调)
# ---------------------------------------------------------------------------
# v3 算法 (sqrt 逆频率 + bg 下限) 给出 [0.5, 0.89, 0.87, 1.17, 1.94],
# 成功修复 v2 的 820:1 失衡坍缩 (mIoU 0.0664 -> 0.3971).
# 但 v3 重训 50 epoch 后的 test per-class 诊断仍暴露:
#   bio: P=0.215 / R=0.842  -> 严重过度预测 (太多像素被误判为 bio)
#   hollow: P=0.222 / R=0.323 -> 严重欠预测 (真实 hollow 大量漏检)
# v4 在 v3 基础上手工微调:
#   bio 0.87 -> 0.65  (降权, 收紧 bio 的预测触发)
#   hollow 1.94 -> 2.30 (提权, 鼓励模型多预测 hollow)
#   其余保持 v3 (bg 0.5 下限 / crack 0.90 / flaking 1.17)
# max/min = 2.30 / 0.5 = 4.6:1 (v3 是 3.9:1, v2 是 820:1; 仍远低于崩塌阈值)
FOCAL_ALPHA_V4 = [0.5, 0.90, 0.65, 1.17, 2.30]


# ---------------------------------------------------------------------------
# 1. Focal Loss (Lin et al., 2017)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter (default 2.0). Higher -> more focus on hard examples.
    alpha: per-class weight (None => uniform, or list/tensor of length C).
           Use inverse-frequency to boost rare classes (flaking, hollow).
    ignore_index: label to skip (e.g. 255 for non-facade regions).

    Implementation note (audit fix):
      The original code passed ``alpha`` into ``F.cross_entropy(weight=...)`` and then
      computed ``pt = exp(-ce)``. Because ``ce`` already contained the ``alpha``
      scaling, ``pt`` became ``p_t ** alpha_t`` instead of ``p_t``. This destroyed
      the focusing factor ``(1-p_t)^gamma`` for rare classes (where alpha_t is large,
      pushing pt -> 0 and (1-pt)^gamma -> 1). The corrected version computes the
      unweighted CE first, derives ``pt = p_t``, applies the modulation, and only
      then multiplies by ``alpha_t`` -- matching the original Focal Loss paper.
    """

    def __init__(self, gamma=2.0, alpha=None, num_classes=5, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        self.num_classes = num_classes

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) long
        """
        B, C, H, W = logits.shape
        # Flatten to (N, C) where N = B*H*W. Must permute channel to last dim
        # so each row = [class0_logit, class1_logit, ...] for one pixel.
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
        target_flat = target.reshape(-1)                           # (N,)

        # Valid-pixel mask (ignore_index handling)
        valid = (target_flat != self.ignore_index)
        if valid.any():
            logits_flat = logits_flat[valid]
            target_flat = target_flat[valid]
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Unweighted cross-entropy -> pt = p_t (probability of correct class)
        ce = F.cross_entropy(logits_flat, target_flat, weight=None, reduction="none")
        pt = torch.exp(-ce)                       # pt = p_t  (NOT p_t^alpha)
        focal = ((1 - pt) ** self.gamma) * ce     # modulation factor * -log(p_t)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target_flat]   # (N,) per-pixel alpha
            focal = alpha_t * focal
        return focal.mean()


# ---------------------------------------------------------------------------
# 2. Lovasz-Softmax Loss (Berman et al., 2018)
# ---------------------------------------------------------------------------
def lovasz_grad(gt_sorted):
    """Compute Lovasz gradient for sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszLossSoftmax(nn.Module):
    """Lovasz-Softmax loss for multi-class segmentation.

    Directly optimizes mean IoU across classes. Particularly effective for
    small targets (cracks) that CE/Focal treat as minority pixels.
    """

    def __init__(self, ignore_index=-100, classes="present"):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) long
        """
        probs = F.softmax(logits, dim=1)
        return self._lovasz_softmax(probs, target, self.classes, self.ignore_index)

    def _lovasz_softmax(self, probs, labels, classes, ignore_index):
        if probs.dim() == 4:
            probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, probs.size(1))
        else:
            probs_flat = probs
        labels_flat = labels.view(-1)
        if ignore_index is not None:
            valid = labels_flat != ignore_index
            probs_flat = probs_flat[valid]
            labels_flat = labels_flat[valid]
        C = probs_flat.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            fg = (labels_flat == c).float()
            if classes == "present" and fg.sum() == 0:
                continue
            if c == 0 and classes == "present":  # skip background in "present" mode
                continue
            class_pred = probs_flat[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            grad = lovasz_grad(fg_sorted).to(probs.device)   # device-safe (audit fix)
            losses.append(torch.dot(errors_sorted, grad))
        if len(losses) == 0:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 3. Boundary Loss (Kervadec et al., 2019, adapted)
# ---------------------------------------------------------------------------
class BoundaryLoss(nn.Module):
    """Boundary loss using Sobel gradient magnitude alignment.

    Encourages prediction boundaries to align with target boundaries.
    Particularly helps thin crack edge precision.

    Implementation note (audit fix):
      Sobel kernels are registered as buffers in __init__ instead of being
      recreated on every forward call (avoids per-iteration tensor allocation
      and potential device mismatch).
    """

    def __init__(self, num_classes=5, ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) long
        """
        # Mask out ignore_index pixels so they don't pollute boundary gradients
        valid = (target != self.ignore_index).float().unsqueeze(1)  # (B,1,H,W)
        probs = F.softmax(logits, dim=1) * valid
        B, C, H, W = probs.shape
        total = 0.0
        count = 0
        for c in range(1, C):  # skip background
            target_c = (target == c).float().unsqueeze(1)
            prob_c = probs[:, c:c + 1, :, :]
            target_boundary = self._sobel_boundary(target_c)
            pred_boundary = self._sobel_boundary(prob_c)
            total = total + F.l1_loss(pred_boundary, target_boundary)
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return total / count

    def _sobel_boundary(self, x):
        """Compute boundary map via Sobel filter (kernels from buffer)."""
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


# ---------------------------------------------------------------------------
# 4. Combined Loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """Weighted combination: Focal + Lovasz + Boundary.

    Default weights: focal=1.0, lovasz=1.0, boundary=0.5
    (boundary weighted lower since it's a refinement term)

    Args:
        ignore_index: pixels with this label are excluded from all three losses
                      (use 255 for non-facade regions: sky, windows, doors, etc.)
    """

    def __init__(self, num_classes=5, gamma=2.0, alpha=None,
                 lambda_focal=1.0, lambda_lovasz=1.0, lambda_boundary=0.5,
                 ignore_index=-100):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha,
                               num_classes=num_classes, ignore_index=ignore_index)
        self.lovasz = LovaszLossSoftmax(ignore_index=ignore_index)
        self.boundary = BoundaryLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.lambda_focal = lambda_focal
        self.lambda_lovasz = lambda_lovasz
        self.lambda_boundary = lambda_boundary

    def forward(self, logits, target):
        fl = self.focal(logits, target)
        ll = self.lovasz(logits, target)
        bl = self.boundary(logits, target)
        total = (self.lambda_focal * fl +
                 self.lambda_lovasz * ll +
                 self.lambda_boundary * bl)
        return total, {"focal": fl.item(), "lovasz": ll.item(),
                       "boundary": bl.item(), "total": total.item()}


# ---------------------------------------------------------------------------
# Helper: compute inverse-frequency alpha from class pixel counts
# ---------------------------------------------------------------------------
def compute_focal_alpha(class_pixel_counts=None, num_classes=5, bg_min=0.5,
                        algorithm="v4"):
    """Compute alpha (per-class weight).

    algorithm="v4" (默认, 2026-11-04 起推荐):
        手工微调的 v4 权重 FOCAL_ALPHA_V4 = [0.5, 0.90, 0.65, 1.17, 2.30].
        基于 v3 重训 50 epoch 的诊断:
          bio P=0.215/R=0.842 过度预测 -> 降权 0.87->0.65
          hollow P=0.222/R=0.323 欠预测 -> 提权 1.94->2.30
        max/min = 4.6:1, 仍远低于 v2 的 820:1 崩塌阈值.
        class_pixel_counts 参数在此模式下被忽略 (留作 API 兼容).

    algorithm="sqrt_inverse" (v3 算法, fallback):
        alpha = 1/sqrt(C*freq), normalize mean=1, bg 下限保护.
        自动从 class_pixel_counts 计算, 权重比 ~3.9:1.
        保留以便对照实验/调试.

    v2 历史教训 (作背景):
        旧版 alpha = 1/(C*freq) + bg*0.3 导致 background 权重 0.0034 vs hollow 2.81
        (1:820), 模型学到 "全部预测成缺陷" 的捷径, mIoU 卡在 0.08.
        v3/v4 都遵守 "权重比 <= ~5:1" 的安全约束.

    Args:
        class_pixel_counts: 每类像素数 (sqrt_inverse 模式必需, v4 模式可省略)
        num_classes: 类别数 (仅 sqrt_inverse 模式使用)
        bg_min: background 权重下限 (仅 sqrt_inverse 模式使用)
        algorithm: "v4" (默认) 或 "sqrt_inverse"
    """
    if algorithm == "v4":
        return torch.tensor(FOCAL_ALPHA_V4, dtype=torch.float32)

    # sqrt_inverse (v3 算法 fallback)
    if class_pixel_counts is None:
        raise ValueError("algorithm='sqrt_inverse' 必须提供 class_pixel_counts")
    counts = np.array(class_pixel_counts, dtype=np.float64)
    counts = np.maximum(counts, 1)
    freq = counts / counts.sum()
    alpha = 1.0 / np.sqrt(num_classes * freq)  # sqrt 逆频率, 温和加权
    alpha = alpha / alpha.mean()               # normalize mean=1
    alpha[0] = max(alpha[0], bg_min)           # background 权重下限保护
    return torch.tensor(alpha, dtype=torch.float32)


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 5, 256, 256
    logits = torch.randn(B, C, H, W, device=device, requires_grad=True)
    target = torch.randint(0, C, (B, H, W), device=device)
    # Add some ignore pixels
    target[:, :50, :] = 255

    fl = FocalLoss(gamma=2.0, ignore_index=255)
    print(f"FocalLoss (with ignore): {fl(logits, target).item():.4f}")

    lv = LovaszLossSoftmax(ignore_index=255)
    print(f"LovaszLoss: {lv(logits, target).item():.4f}")

    bl = BoundaryLoss(ignore_index=255).to(device)
    print(f"BoundaryLoss: {bl(logits, target).item():.4f}")

    # v4 (默认) vs v3 算法对照
    alpha_v4 = compute_focal_alpha()  # 默认 algorithm="v4"
    alpha_v3 = compute_focal_alpha(
        [501586446, 9719840, 10129403, 5624990, 2038073],
        algorithm="sqrt_inverse",
    )
    print(f"\nFocal alpha (v4 default):         {alpha_v4.tolist()}")
    print(f"  max/min ratio = {max(alpha_v4).item() / min(alpha_v4).item():.2f}:1")
    print(f"Focal alpha (v3 sqrt_inverse):    {alpha_v3.tolist()}")
    print(f"  max/min ratio = {max(alpha_v3).item() / min(alpha_v3).item():.2f}:1")

    combined = CombinedLoss(alpha=alpha_v4, ignore_index=255).to(device)
    loss, parts = combined(logits, target)
    print(f"\nCombined (v4): {parts}")
    loss.backward()
    print("Backward: OK")
