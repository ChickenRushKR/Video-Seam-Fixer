"""Wan-VACE first-last-frame-to-video (flf2v) bridge generation.

VACE generates the frames BETWEEN two fixed keyframes (first + last). Used for
length-preserving boundary bridging: the keyframes are the real frames just
outside the replaced window, and VACE fills the middle. Stronger inbetweening
model than Wan2.2 I2V; conditions on BOTH endpoints natively.

Caveats: Wan2.1-VACE-1.3B is a 480p model (upscaling to native may look soft),
and the middle motion is free-generated (no explicit motion control) — so for
choreography-critical content the generated motion may still diverge.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _to_pil(frame: torch.Tensor, size_wh: tuple[int, int]):
    from PIL import Image

    f = F.interpolate(frame.unsqueeze(0).float().clamp(0, 1), size=(size_wh[1], size_wh[0]),
                      mode="bilinear", align_corners=False)[0]
    return Image.fromarray((f.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def _frames_to_tensor(frames) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(np.asarray(f).astype(np.float32) / 255.0).permute(2, 0, 1) for f in frames]
    )


class VaceGenerator:
    _pipe = None

    def __init__(self, model_id="Wan-AI/Wan2.1-VACE-1.3B-diffusers", dtype="bfloat16",
                 cpu_offload=True, flow_shift=3.0, device="cuda", quantize="none"):
        if VaceGenerator._pipe is None:
            from diffusers import AutoencoderKLWan, WanVACEPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            kwargs = dict(vae=vae, torch_dtype=_DTYPE[dtype])

            quantized = quantize and quantize != "none"
            if quantized:
                # Quantize the (huge) transformer so the 14B fits in VRAM without
                # cpu-offload thrashing (offload made it ~43 min/step).
                from diffusers import TorchAoConfig, WanVACETransformer3DModel
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig,
                    Float8WeightOnlyConfig,
                    Int8WeightOnlyConfig,
                )

                ao = {
                    "int8wo": Int8WeightOnlyConfig,
                    "float8wo": Float8WeightOnlyConfig,
                    "float8dq": Float8DynamicActivationFloat8WeightConfig,
                }[quantize]()
                tr = WanVACETransformer3DModel.from_pretrained(
                    model_id, subfolder="transformer",
                    quantization_config=TorchAoConfig(ao), torch_dtype=_DTYPE[dtype],
                )
                kwargs["transformer"] = tr

            pipe = WanVACEPipeline.from_pretrained(model_id, **kwargs)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
            if cpu_offload:
                # With int8 the transformer is ~14GB; offload keeps only the ACTIVE
                # module resident (text-encoder offloaded after prompt encoding), so
                # the transformer + 720p activations fit without spilling -> fast.
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            VaceGenerator._pipe = pipe
        self.pipe = VaceGenerator._pipe

    @torch.no_grad()
    def flf2v(self, first: torch.Tensor, last: torch.Tensor, num_frames: int,
              height=832, width=480, steps=30, guidance=5.0, prompt="", negative_prompt="",
              seed=0) -> torch.Tensor:
        from PIL import Image

        fp = _to_pil(first, (width, height))
        lp = _to_pil(last, (width, height))
        gray = Image.new("RGB", (width, height), (128, 128, 128))
        black = Image.new("L", (width, height), 0)
        white = Image.new("L", (width, height), 255)
        video = [fp] + [gray] * (num_frames - 2) + [lp]
        mask = [black] + [white] * (num_frames - 2) + [black]

        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt or "smooth natural continuation, same scene and subject",
            negative_prompt=negative_prompt,
            video=video, mask=mask, reference_images=[fp],
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=steps, guidance_scale=guidance, generator=gen, output_type="pil",
        )
        return _frames_to_tensor(out.frames[0])  # [N,3,H,W]
