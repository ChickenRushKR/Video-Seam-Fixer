"""Wan2.2 image-to-video generation harness (canvas-aligned clip continuation).

Goal (Branch: generation-time canvas alignment): generate clip N+1 conditioned on
clip N's LAST frame. ``WanImageToVideoPipeline`` pins the output's first frame to
the conditioning image, so the new clip starts exactly where the previous one
ended — same canvas/scale/framing — eliminating the boundary scale ("비율") drift
at generation time instead of post-hoc.

Chaining clips autoregressively (each conditioned on the previous clip's last
generated frame) keeps the canvas continuous across an arbitrary sequence.

NOTE: base Wan2.2-TI2V-5B is image(+text)-to-video; it does NOT take 3D-motion
control. So the *motion* it generates is its own, not the source choreography —
motion-faithful generation needs a control model (e.g. Wan-VACE) as a later layer.
This harness establishes the canvas-alignment mechanism.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _to_pil(frame: torch.Tensor, size_wh: tuple[int, int]):
    """frame [3,H,W] float in [0,1] -> PIL resized to (W, H)."""
    from PIL import Image

    f = frame.detach().float().clamp(0, 1).unsqueeze(0)
    f = F.interpolate(f, size=(size_wh[1], size_wh[0]), mode="bilinear", align_corners=False)[0]
    arr = (f.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _frames_to_tensor(frames) -> torch.Tensor:
    out = [torch.from_numpy(np.asarray(f).astype(np.float32) / 255.0).permute(2, 0, 1) for f in frames]
    return torch.stack(out, dim=0)  # [N,3,H,W] in [0,1]


class WanGenerator:
    """Lazy-loaded Wan2.2 I2V pipeline for canvas-aligned continuation."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        dtype: str = "bfloat16",
        cpu_offload: bool = True,
        device: str = "cuda",
    ):
        from diffusers import WanImageToVideoPipeline

        pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=_DTYPE[dtype])
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        self.pipe = pipe
        self.device = device

    @torch.no_grad()
    def continue_from(
        self,
        cond_frame: torch.Tensor,        # [3,H,W] float [0,1] — previous clip's last frame
        num_frames: int = 49,            # must be 4k+1 (Wan rounds otherwise)
        height: int = 1280,
        width: int = 720,
        steps: int = 30,
        guidance: float = 5.0,
        prompt: str = "smooth natural continuation of the same scene",
        seed: int = 0,
        last_frame: torch.Tensor | None = None,  # optional end keyframe (two-keyframe mode)
    ) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        kwargs = dict(
            image=_to_pil(cond_frame, (width, height)),
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            output_type="pil",
        )
        if last_frame is not None:
            kwargs["last_image"] = _to_pil(last_frame, (width, height))
        result = self.pipe(**kwargs)
        return _frames_to_tensor(result.frames[0])  # [N,3,H,W]
