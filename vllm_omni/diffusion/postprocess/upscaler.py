# SPDX-License-Identifier: Apache-2.0
# Post-processing video upscaler (Real-ESRGAN RRDBNet) for vLLM-Omni.
#
# Design constraints:
#   * no dependency on basicsr / realesrgan (absent in the image, and we should not add it)
#   * input/output stay BCTHW + NORMALIZED_FLOAT, so it can be plugged into the
#     float-consumer pipeline of finalize_diffusion_media()
#   * tiling + feathered overlap keeps RRDB feature maps from blowing up VRAM at 768p -> 4x
#   * output follows `scale`; when native scale != target scale, fall back with area interpolation

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CACHE: dict[tuple[str, str], "RRDBNet"] = {}

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
    """Matches basicsr/RealESRGAN RRDB exactly: only rdb1..rdb3, no conv_body."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN backbone, key names aligned with the official RealESRGAN_x4plus / x2 weights."""

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
    """1-D feather weights: linear ramp on edges that have a neighbouring tile."""
    w = torch.ones(length)
    overlap = min(overlap, max(1, length // 2))
    if edges[0]:
        w[:overlap] = torch.linspace(1.0 / (overlap + 1), 1.0, overlap)
    if edges[1]:
        w[length - overlap:] = torch.linspace(1.0, 1.0 / (overlap + 1), overlap)
    return w


def _upscale_one(net: RRDBNet, frame: torch.Tensor, native_scale: int, tile: int, overlap: int) -> torch.Tensor:
    """frame: 1,3,H,W in [0,1]; returns 1,3,H*s,W*s with the network weight dtype."""
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
    """
    if not model_path:
        raise ValueError("upscale_model_path is required when upscale is enabled")
    if tensor.ndim != 5:
        raise ValueError(f"Expected BCTHW tensor, got shape {tuple(tensor.shape)}")

    device = tensor.device
    dtype = tensor.dtype
    net = _load_net(model_path, device, native_scale, num_block, half)

    negative_one = float(tensor.min()) < -0.05

    batch, channels, frames, height, width = tensor.shape
    out_h, out_w = int(round(height * scale)), int(round(width * scale))
    out = torch.empty((batch, channels, frames, out_h, out_w), device=device, dtype=dtype)

    with torch.no_grad():
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
