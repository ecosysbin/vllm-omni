# SPDX-License-Identifier: Apache-2.0
# Post-processing video upscaler (Real-ESRGAN RRDBNet) for vLLM-Omni.
# vLLM-Omni 的视频后处理超分器（Real-ESRGAN RRDBNet）。
#
# Design constraints:
#   * no dependency on basicsr / realesrgan (absent in the image, and we should not add it)
#   * input/output stay BCTHW + NORMALIZED_FLOAT, so it can be plugged into the
#     float-consumer pipeline of finalize_diffusion_media()
#   * tiling + feathered overlap keeps RRDB feature maps from blowing up VRAM at 768p -> 4x
#   * output follows `scale`; when native scale != target scale, fall back with area interpolation
# 设计约束：
#   * 不依赖 basicsr / realesrgan（镜像里没有，也不应该为此新增依赖）
#   * 输入输出保持 BCTHW + NORMALIZED_FLOAT，以便接入 finalize_diffusion_media()
#     的浮点消费者管线
#   * 分块推理 + 羽化重叠，避免 RRDB 特征图在 768p 放大时把显存打爆
#   * 输出倍数由 `scale` 决定；当权重原生倍数与目标倍数不一致时，用 area 插值回落到目标尺寸

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 已加载的网络按（权重路径、设备、精度、原生倍数、block 数）缓存，避免每帧重复加载。
_CACHE: dict[tuple[str, str], "RRDBNet"] = {}

# 相邻分块之间保留的重叠像素，用于羽化拼接、消除接缝。
_OVERLAP = 16


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Matches basicsr/RealESRGAN RRDB exactly: only rdb1..rdb3, no conv_body.

    与 basicsr/RealESRGAN 的 RRDB 结构完全一致：只有 rdb1..rdb3，没有 conv_body。
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN backbone, key names aligned with the official RealESRGAN_x4plus / x2 weights.

    Real-ESRGAN 主干网络，参数命名与官方 RealESRGAN_x4plus / x2 权重对齐。
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # The official x2 weights (RealESRGAN_x2plus) only carry one upsample stage,
        # so scale=2 must not register conv_up2 or strict loading fails on missing keys.
        # 官方 x2 权重（RealESRGAN_x2plus）只有一级上采样，
        # 因此 scale=2 时不能注册 conv_up2，否则严格加载会因缺 key 失败。
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) if scale != 2 else None
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.conv_up2 is not None:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _load_net(model_path: str, device: torch.device, native_scale: int, num_block: int, half: bool = False) -> RRDBNet:
    # native_scale/num_block change the module structure, so they belong in the cache key;
    # otherwise a scale=4 net could be reused for a scale=2 request on the same path.
    # native_scale / num_block 会改变模块结构，所以必须进缓存 key；
    # 否则同一路径下 scale=4 的网络会被 scale=2 的请求复用，权重根本对不上。
    key = (model_path, str(device), str(half), int(native_scale), int(num_block))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Upscaler weights must be a state dict mapping: {model_path}")
    sd = state.get("params_ema") or state.get("params") or state
    # Real-ESRGAN x2 weights (RealESRGAN_x2plus) take a pixel-unshuffled 12-channel input
    # (3 x 2x2) and still run two 2x stages internally: the heavy RRDB body therefore works
    # on a quarter of the pixels and the output lands at 2x of the original frame. See
    # "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data".
    # 官方 x2 权重（RealESRGAN_x2plus）接收像素解混后的 12 通道输入（3 x 2x2），
    # 内部仍然跑两级 2x 上采样：繁重的 RRDB 主体因此在四分之一的像素上计算，
    # 输出正好是原帧的 2 倍。参见论文《Real-ESRGAN: Training Real-World Blind
    # Super-Resolution with Pure Synthetic Data》。
    num_in_ch = 12 if native_scale == 2 else 3
    net = RRDBNet(num_in_ch=num_in_ch, scale=4, num_block=num_block)
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    if half:
        net.half()
    for p in net.parameters():
        p.requires_grad_(False)
    _CACHE[key] = net
    return net


def _axis_weights(length: int, overlap: int, edges: tuple[bool, bool]) -> torch.Tensor:
    """1-D feather weights: linear ramp on edges that have a neighbouring tile.

    一维羽化权重：只在有相邻分块的那条边上做线性斜坡，避免拼接缝。
    """
    w = torch.ones(length)
    overlap = min(overlap, max(1, length // 2))
    if edges[0]:
        w[:overlap] = torch.linspace(1.0 / (overlap + 1), 1.0, overlap)
    if edges[1]:
        w[length - overlap:] = torch.linspace(1.0, 1.0 / (overlap + 1), overlap)
    return w


def _upscale_one(net: RRDBNet, frame: torch.Tensor, native_scale: int, tile: int, overlap: int) -> torch.Tensor:
    """frame: 1,3,H,W in [0,1]; returns 1,3,H*s,W*s with the network weight dtype.

    输入 1,3,H,W、取值 [0,1] 的单帧；返回 1,3,H*s,W*s，dtype 与网络权重一致。

    按 tile 切块推理，每块乘羽化掩码后累加，最后除以权重和，得到无缝的整帧。
    """
    dtype = next(net.parameters()).dtype
    frame = frame.to(dtype)
    _, _, height, width = frame.shape
    out_h, out_w = height * native_scale, width * native_scale
    acc = torch.zeros((1, 3, out_h, out_w), device=frame.device, dtype=dtype)
    weight = torch.zeros((1, 1, out_h, out_w), device=frame.device, dtype=dtype)

    tile = max(64, min(tile, max(height, width)))
    step = max(32, tile - overlap)
    for y0 in range(0, height, step):
        y1 = min(y0 + tile, height)
        for x0 in range(0, width, step):
            x1 = min(x0 + tile, width)
            patch = frame[..., y0:y1, x0:x1]
            ph, pw = y1 - y0, x1 - x0
            if ph % 2 or pw % 2:
                # pixel_unshuffle needs even spatial dims; replicate-pad then crop the result.
                # pixel_unshuffle 要求空间尺寸为偶数；先复制填充，输出时再裁回原尺寸。
                patch = F.pad(patch, (0, pw % 2, 0, ph % 2), mode="replicate")
            if native_scale == 2:
                patch = F.pixel_unshuffle(patch, 2)
            out = net(patch)
            out = out[..., : ph * native_scale, : pw * native_scale]
            oy0, ox0 = y0 * native_scale, x0 * native_scale
            oy1, ox1 = oy0 + out.shape[-2], ox0 + out.shape[-1]
            wy = _axis_weights(out.shape[-2], overlap * native_scale, (y0 > 0, y1 < height)).to(dtype)
            wx = _axis_weights(out.shape[-1], overlap * native_scale, (x0 > 0, x1 < width)).to(dtype)
            mask = (wy[:, None] * wx[None, :]).to(out.device)[None, None]
            acc[..., oy0:oy1, ox0:ox1] += out * mask
            weight[..., oy0:oy1, ox0:ox1] += mask
    return acc / weight.clamp_min(torch.finfo(dtype).eps)


def upscale_video_tensor(
    tensor: torch.Tensor,
    *,
    scale: float = 2.0,
    model_path: str | None = None,
    tile: int = 256,
    max_frames: int = 8,
    native_scale: int = 4,
    num_block: int = 23,
    half: bool = False,
) -> torch.Tensor:
    """Upscale a BCTHW normalized-float video tensor.

    scale: target factor (default 2 turns 1344x768 into 2688x1536)
    native_scale: native factor of the weights (RealESRGAN_x4plus=4, x2 variant=2)
    half: run inference in fp16 (roughly 1.5-2x faster and lower VRAM)

    把 BCTHW 归一化浮点视频张量整体超分。

    scale: 目标倍数（默认 2，即 1344x768 变成 2688x1536）
    native_scale: 权重原生倍数（RealESRGAN_x4plus=4，x2 变体=2）
    half: 用 fp16 推理（约快 1.5-2 倍，显存也更省）
    """
    if not model_path:
        raise ValueError("upscale_model_path is required when upscale is enabled")
    if tensor.ndim != 5:
        raise ValueError(f"Expected BCTHW tensor, got shape {tuple(tensor.shape)}")

    device = tensor.device
    dtype = tensor.dtype
    net = _load_net(model_path, device, native_scale, num_block, half)

    # 自动识别值域：出现负值就按 [-1,1] 处理，先映射到 [0,1] 推理，最后再映射回去。
    negative_one = float(tensor.min()) < -0.05

    batch, channels, frames, height, width = tensor.shape
    out_h, out_w = int(round(height * scale)), int(round(width * scale))
    out = torch.empty((batch, channels, frames, out_h, out_w), device=device, dtype=dtype)

    with torch.no_grad():
        # 按 max_frames 分段逐帧推理，控制同一时刻的显存峰值。
        chunk = max(1, int(max_frames))
        for start in range(0, frames, chunk):
            end = min(start + chunk, frames)
            for b in range(batch):
                for t in range(start, end):
                    frame = tensor[b, :, t].unsqueeze(0).float()
                    if negative_one:
                        frame = frame.mul(0.5).add(0.5)
                    frame = frame.clamp_(0.0, 1.0)
                    up = _upscale_one(net, frame, native_scale, tile, _OVERLAP)
                    # 权重原生倍数与目标倍数不一致时，用 area 插值缩到目标尺寸
                    # （比双线性更不容易在降采样时产生振铃）。
                    if native_scale != scale:
                        up = F.interpolate(up, size=(out_h, out_w), mode="area")
                    if negative_one:
                        up = up.mul(2.0).sub(1.0)
                    out[b, :, t] = up.squeeze(0).to(dtype)
    return out


def upscale_numpy_videos(
    videos,
    *,
    scale: float = 2.0,
    model_path: str | None = None,
    tile: int = 512,
    max_frames: int = 8,
    native_scale: int = 4,
    num_block: int = 23,
    half: bool = True,
    device: str | None = None,
    logger: object | None = None,
):
    """Serving-layer entry point: upscale a list of numpy frame arrays (THWC/TCHW uint8).

    Streaming per frame: only one frame lives on the device at a time, because a whole-clip
    4x/2x buffer would OOM (H3 keeps the card nearly full after sampling).

    Non-4D inputs pass through untouched; exceptions are handled (degraded) by the caller.

    服务层入口：对一组 numpy 帧数组（THWC 或 TCHW、uint8）做超分。

    逐帧流水处理：同一时刻只有一帧在设备上，因为整段 4x/2x 的缓冲会 OOM
    （H3 采样结束后显存本来就已经接近打满）。

    非 4 维输入原样返回；异常由调用方兜底（降级为基础帧）。
    """
    import numpy as np

    if not model_path:
        raise ValueError("upscale_model_path is required when upscale is enabled")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = _load_net(model_path, dev, native_scale, num_block, half)

    out_list = []
    for item in videos:
        if not isinstance(item, np.ndarray) or item.ndim != 4:
            out_list.append(item)
            continue
        if item.shape[-1] == 3:
            layout = "thwc"
            frames_t, height, width, _ = item.shape
        elif item.shape[1] == 3:
            layout = "tchw"
            frames_t, _, height, width = item.shape
        else:
            out_list.append(item)
            continue

        out_h, out_w = int(round(height * scale)), int(round(width * scale))
        if logger is not None:
            logger.info("video upscale start (serving layer): %s -> %s", item.shape, layout)
        out_np = np.empty((frames_t, out_h, out_w, 3), dtype=np.uint8)
        for t in range(frames_t):
            src = item[t][..., :3] if layout == "thwc" else item[t, :3]
            src = np.ascontiguousarray(src)
            frame = torch.from_numpy(src).to(dev)
            if not frame.is_floating_point():
                frame = frame.float().div_(255.0)
            else:
                frame = frame.float()
            frame = frame.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
            frame = frame.clamp_(0.0, 1.0)
            with torch.inference_mode():
                up = _upscale_one(net, frame, native_scale, tile, _OVERLAP)
                if native_scale != scale:
                    up = F.interpolate(up, size=(out_h, out_w), mode="area")
            up = up[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            out_np[t] = up.permute(1, 2, 0).contiguous().cpu().numpy()
            del frame, up
        arr = out_np if layout == "thwc" else np.ascontiguousarray(np.transpose(out_np, (0, 3, 1, 2)))
        out_list.append(arr)
        if logger is not None:
            logger.info("video upscale done (serving layer): %s", arr.shape)
        del out_np
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out_list
