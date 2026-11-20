# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import replace

from vllm.logger import init_logger

from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    FloatVideoConsumer,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTransportConstraints,
    VideoValueRange,
)
from vllm_omni.diffusion.postprocess.rife_interpolator import interpolate_video_tensor
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


logger = init_logger(__name__)


def finalize_diffusion_media(
    media: DiffusionMediaOutput,
    *,
    sampling_params: OmniDiffusionSamplingParams,
) -> dict[str, object]:
    """Finalize typed video media without invoking a model postprocessor."""
    media.validate()
    if not media.prepared_for_transport:
        raise ValueError("Diffusion media reached the engine before transport preparation")

    video = media.video
    metadata: dict[str, object] = {}
    consumers = video.constraints.pending_float_consumers
    if FloatVideoConsumer.FRAME_INTERPOLATION in consumers:
        if video.spec.encoding is not VideoTensorEncoding.NORMALIZED_FLOAT:
            raise ValueError("Frame interpolation requires NORMALIZED_FLOAT video")
        # RIFE guesses [0,1] vs [-1,1] from the tensor's min/max, so relying on
        # the sample content misclassifies both an all-nonnegative [-1,1] clip
        # (read as [0,1]) and a [0,1] clip whose single bf16 value overshoots 1
        # (read as [-1,1]). Always map to unit range per the declared spec and
        # clamp to [0,1] before RIFE, then restore the declared range so
        # downstream denormalization is correct regardless of the sample.
        to_negative_one = video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE
        if to_negative_one:
            interp_input = video.tensor.mul(0.5).add(0.5).clamp_(0.0, 1.0)
        else:
            interp_input = video.tensor.clamp(0.0, 1.0)
        interpolated, multiplier = interpolate_video_tensor(
            interp_input,
            exp=sampling_params.frame_interpolation_exp,
            scale=sampling_params.frame_interpolation_scale,
            model_path=sampling_params.frame_interpolation_model_path,
        )
        if to_negative_one:
            interpolated = interpolated.mul(2.0).sub(1.0)
        consumers = consumers - {FloatVideoConsumer.FRAME_INTERPOLATION}
        video = replace(
            video,
            tensor=interpolated,
            constraints=VideoTransportConstraints(pending_float_consumers=frozenset(consumers)),
        )
        metadata["video"] = {"video_fps_multiplier": multiplier}

    # 超分消费者在这里按采样参数登记（约束有可能在这一步之前已被丢弃；
    # 同时兼容 uint8 帧输入）。失败时降级为原始帧，并记录原因。
    if getattr(sampling_params, "enable_upscale", False):
        consumers = consumers | {FloatVideoConsumer.UPSCALE}
    if FloatVideoConsumer.UPSCALE in consumers:
        try:
            from vllm_omni.diffusion.media import VideoTensorSpec
            from vllm_omni.diffusion.postprocess.upscaler import upscale_video_tensor

            logger.info(
                "video upscale start: encoding=%s shape=%s scale=%s path=%s",
                video.spec.encoding,
                tuple(video.tensor.shape),
                float(getattr(sampling_params, "upscale_scale", 2.0)),
                getattr(sampling_params, "upscale_model_path", None),
            )
            if video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES:
                if video.spec.layout is not VideoTensorLayout.BTHWC:
                    raise ValueError("UINT8_FRAMES video must use BTHWC layout for upscale")
                tensor = video.tensor.float().div_(255.0).permute(0, 4, 1, 2, 3).contiguous()
                video = replace(
                    video,
                    tensor=tensor,
                    spec=VideoTensorSpec(
                        layout=VideoTensorLayout.BCTHW,
                        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                        value_range=VideoValueRange.ZERO_TO_ONE,
                    ),
                )
            # 其余编码（例如已压成 uint8 且不是 BTHWC）无法直接交给 RRDB，直接报错交给兜底。
            elif video.spec.encoding is not VideoTensorEncoding.NORMALIZED_FLOAT:
                raise ValueError(f"Upscale does not support encoding={video.spec.encoding}")
            upscaled = upscale_video_tensor(
                video.tensor,
                scale=float(getattr(sampling_params, "upscale_scale", 2.0)),
                model_path=getattr(sampling_params, "upscale_model_path", None),
                native_scale=int(getattr(sampling_params, "upscale_native_scale", 4)),
                tile=int(getattr(sampling_params, "upscale_tile", 256)),
                max_frames=int(getattr(sampling_params, "upscale_max_frames", 8)),
            )
            consumers = consumers - {FloatVideoConsumer.UPSCALE}
            video = replace(
                video,
                tensor=upscaled,
                constraints=VideoTransportConstraints(pending_float_consumers=frozenset(consumers)),
            )
            metadata.setdefault("video", {})["upscale"] = float(
                getattr(sampling_params, "upscale_scale", 2.0)
            )
            logger.info("video upscale done: shape=%s", tuple(upscaled.shape))
        except Exception as exc:  # fallback to base frames
            logger.warning("video upscale failed, falling back to base frames: %s", exc)
            consumers = consumers - {FloatVideoConsumer.UPSCALE}
            video = replace(
                video,
                constraints=VideoTransportConstraints(pending_float_consumers=frozenset(consumers)),
            )
            metadata.setdefault("video", {})["upscale_fallback"] = str(exc)[:200]

    if consumers:
        names = sorted(consumer.value for consumer in consumers)
        raise ValueError(f"Video has pending float consumers at finalization: {names}")

    output_type = getattr(sampling_params, "output_type", None) or "np"
    if output_type == "latent":
        raise ValueError("Latent output cannot be finalized from VideoMediaOutput")

    if video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES:
        if video.spec.layout is not VideoTensorLayout.BTHWC:
            raise ValueError("UINT8_FRAMES video must use BTHWC layout")
        if output_type != "np":
            raise ValueError(f"UINT8_FRAMES finalization does not support output_type={output_type!r}")
        frames = video.tensor.detach().cpu().numpy()
    else:
        if video.spec.layout is not VideoTensorLayout.BCTHW:
            raise ValueError("NORMALIZED_FLOAT video must use BCTHW layout")
        from diffusers.video_processor import VideoProcessor

        do_denormalize = video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE
        frames = VideoProcessor(vae_scale_factor=8).postprocess_video(
            video.tensor,
            output_type=output_type,
            do_denormalize=[do_denormalize] * video.tensor.shape[2],
        )

    return {
        "payload": {"video": frames},
        "metadata": metadata,
    }
