# -*- coding: utf-8 -*-
"""
滑窗推理工具 (train.py / inference.py 共享)
=============================================
审计修复 N1+N2:
  - N1: 原滑窗 range(0, Hh-crop+1, stride) 在右/下边缘有覆盖洞
        (1088×1456 图: 右 176 列 + 下 64 行永远未覆盖, argmax 全背景).
        修复: _patch_starts() 保证最后一个 patch 起点在 size-crop, 覆盖到底/右.
  - N2: inference 原整图推理无 autocast, OS=16 下可能 OOM.
        修复: 统一用滑窗 + autocast, TTA 按 patch 做.
"""
import torch
import torch.nn.functional as F


def _patch_starts(size, crop, stride):
    """生成 patch 起始坐标, 保证覆盖到边缘.

    例: size=1088, crop=512, stride=256
        原始 range: [0, 256, 512] -> 覆盖到 1024, 漏 1024-1088
        修复后:     [0, 256, 512, 576] -> 最后 patch 覆盖 576-1088, 全覆盖
    """
    if size <= crop:
        return [0]
    starts = list(range(0, size - crop + 1, stride))
    # 确保最后一个 patch 起点在 size-crop, 覆盖到最右/下边缘
    if starts[-1] != size - crop:
        starts.append(size - crop)
    return starts


@torch.no_grad()
def sliding_window_predict(model, image_tensor, device,
                           crop=512, stride=256, tta=False):
    """滑窗推理整图, 返回完整概率图 (C, H, W).

    Args:
        image_tensor: (C_in, H, W) normalized tensor on CPU
        crop: patch 大小 (默认 512)
        stride: 滑窗步长 (默认 256, 重叠 50%)
        tta: 是否做水平翻转 TTA (每个 patch 翻转推理后翻回取平均)
    Returns: (num_classes, H, W) softmax probability map on CPU
    """
    num_classes = 5
    _, Hh, Ww = image_tensor.shape

    # 图像小于 crop 时直接整图推理
    if Hh <= crop and Ww <= crop:
        inp = image_tensor.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(inp)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        if tta:
            inp_flip = torch.flip(image_tensor, dims=[2]).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits_flip = model(inp_flip)
            probs_flip = F.softmax(logits_flip, dim=1).squeeze(0).cpu()
            probs = (probs + torch.flip(probs_flip, dims=[2])) / 2.0
        return probs

    # 如果图像小于 crop, pad 到 crop
    pad_h = max(0, crop - Hh)
    pad_w = max(0, crop - Ww)
    if pad_h or pad_w:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        _, Hh, Ww = image_tensor.shape

    prob_sum = torch.zeros(num_classes, Hh, Ww)
    count = torch.zeros(1, Hh, Ww)

    y_starts = _patch_starts(Hh, crop, stride)
    x_starts = _patch_starts(Ww, crop, stride)

    for y0 in y_starts:
        for x0 in x_starts:
            patch = image_tensor[:, y0:y0+crop, x0:x0+crop]
            inp = patch.unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(inp)
            prob = F.softmax(logits.float(), dim=1).squeeze(0).cpu()
            prob_sum[:, y0:y0+crop, x0:x0+crop] += prob
            count[:, y0:y0+crop, x0:x0+crop] += 1

            if tta:
                inp_flip = torch.flip(inp, dims=[3])
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits_flip = model(inp_flip)
                prob_flip = F.softmax(logits_flip.float(), dim=1).squeeze(0).cpu()
                prob_flip = torch.flip(prob_flip, dims=[2])
                prob_sum[:, y0:y0+crop, x0:x0+crop] += prob_flip
                count[:, y0:y0+crop, x0:x0+crop] += 1

    avg_prob = prob_sum / count.clamp(min=1)
    # 裁回原始尺寸 (去掉 pad)
    orig_h = Hh - pad_h if pad_h else Hh
    orig_w = Ww - pad_w if pad_w else Ww
    return avg_prob[:, :orig_h, :orig_w]


if __name__ == "__main__":
    # 测试 _patch_starts
    print("=== N1 验证: _patch_starts 覆盖边缘 ===")
    ys = _patch_starts(1088, 512, 256)
    xs = _patch_starts(1456, 512, 256)
    print(f"Y starts: {ys} -> last patch covers {ys[-1]}:{ys[-1]+512}, "
          f"bottom edge {ys[-1]+512} vs 1088: {'COVERED' if ys[-1]+512 >= 1088 else 'GAP!'}")
    print(f"X starts: {xs} -> last patch covers {xs[-1]}:{xs[-1]+512}, "
          f"right edge {xs[-1]+512} vs 1456: {'COVERED' if xs[-1]+512 >= 1456 else 'GAP!'}")
    assert ys[-1] + 512 >= 1088, "Y edge not covered!"
    assert xs[-1] + 512 >= 1456, "X edge not covered!"
    print("PASS: 全图覆盖, 无边缘洞")
