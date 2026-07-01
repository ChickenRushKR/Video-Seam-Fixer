"""GenerativeRefiner — diffusion video transition synthesis (Branch 1).

The residual seam difference that geometric warps cannot fix is *content-level*
(the generative model produced B with locally different content than a true
continuation of A). This backend *synthesizes* the transition instead of moving
pixels.

Default: Wan2.2-TI2V-5B via ``WanImageToVideoPipeline`` with TWO-keyframe
conditioning — ``image`` = A's last frame, ``last_image`` = B's boundary frame —
so the model generates a content-aware morph from A into B (not a cross-fade).
The synthesized frames replace B's first ``num_frames`` frames
(``ReplaceCorrection``), bridging the clips with model-coherent content.

The pipeline class / model are configurable; any diffusers video pipeline whose
``__call__`` takes ``image`` (and optionally ``last_image``) works.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vbf.data.window import BoundaryWindow
from vbf.refiners.base import (
    BoundaryRefiner,
    BridgeReplaceCorrection,
    Correction,
    register_refiner,
)

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _to_pil(frame: torch.Tensor, size_wh: tuple[int, int]):
    """frame [3,H,W] float in [0,1] -> PIL resized to (W, H)."""
    from PIL import Image

    f = frame.detach().float().clamp(0, 1).unsqueeze(0)
    f = F.interpolate(f, size=(size_wh[1], size_wh[0]), mode="bilinear", align_corners=False)[0]
    arr = (f.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _hist_match_channel(src: torch.Tensor, ref: torch.Tensor, n: int = 256) -> torch.Tensor:
    """Map src values through ref's distribution (per-channel histogram/LUT match)."""
    qs = torch.linspace(0, 1, n, device=src.device, dtype=src.dtype)
    # subsample for stable, cheap quantiles
    def sub(x):
        return x if x.numel() <= 200000 else x[torch.randint(0, x.numel(), (200000,), device=x.device)]
    src_q = torch.quantile(sub(src), qs)
    ref_q = torch.quantile(sub(ref), qs)
    idx = torch.searchsorted(src_q.contiguous(), src.clamp(float(src_q[0]), float(src_q[-1]))).clamp(0, n - 1)
    return ref_q[idx]


def _domain_match(gen: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reduce the generated<->real domain gap: per-channel histogram (colour) match
    + grain/noise level match. Structural/proportion differences are NOT fixable here."""
    out = gen.clone()
    for c in range(3):
        out[:, c] = _hist_match_channel(gen[:, c].reshape(-1), ref[:, c].reshape(-1)).reshape(gen[:, c].shape)
    # grain match: bring generated high-frequency noise up to the real footage level
    def hi(x):
        return x - F.avg_pool2d(x, 3, stride=1, padding=1)
    ref_n = float(hi(ref).std())
    gen_n = float(hi(out).std())
    if ref_n > gen_n:
        add = (ref_n ** 2 - gen_n ** 2) ** 0.5
        out = out + torch.randn_like(out) * add
    return out.clamp(0, 1)


def _color_match(frames: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Match per-channel mean/std of ``frames`` [N,3,H,W] to reference frames ``ref`` [K,3,H,W].

    Reduces the colour/exposure domain gap between generated and original frames,
    so the generated↔real splice does not pop.
    """
    eps = 1e-5
    tgt = ref.flatten(2).permute(1, 0, 2).reshape(3, -1)   # [3, K*H*W]
    src = frames.flatten(2).permute(1, 0, 2).reshape(3, -1)  # [3, N*H*W]
    tm, ts = tgt.mean(1), tgt.std(1)
    sm, ss = src.mean(1), src.std(1)
    gain = (ts / (ss + eps)).clamp(0.5, 2.0).view(1, 3, 1, 1)
    bias = (tm.view(1, 3, 1, 1) - gain * sm.view(1, 3, 1, 1))
    return (frames * gain + bias).clamp(0, 1)


def _frames_to_tensor(frames) -> torch.Tensor:
    """diffusers frame list (PIL) -> [N, 3, H, W] float in [0,1]."""
    out = []
    for f in frames:
        a = torch.from_numpy(np.asarray(f).astype(np.float32) / 255.0)
        out.append(a.permute(2, 0, 1))
    return torch.stack(out, dim=0)


@register_refiner("generative")
class GenerativeRefiner(BoundaryRefiner):
    _pipe = None  # class-level cache (avoid reloading the multi-GB model per run)

    def _pipeline(self):
        if GenerativeRefiner._pipe is None:
            import diffusers

            gcfg = self.config.refiner.generative
            cls = getattr(diffusers, gcfg.pipeline_class)
            pipe = cls.from_pretrained(gcfg.model_id, torch_dtype=_DTYPE[gcfg.dtype])
            if gcfg.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            GenerativeRefiner._pipe = pipe
        return GenerativeRefiner._pipe

    def refine(self, window: BoundaryWindow) -> tuple[BoundaryWindow, Correction]:
        gcfg = self.config.refiner.generative
        window.to(self.device)
        size_wh = (gcfg.width, gcfg.height)

        a_keep, b_keep = gcfg.a_keep, gcfg.b_keep
        a_repl = a_keep - 1                        # A frames replaced (A's last a_repl)
        b_repl = b_keep                            # B frames replaced (B's first b_repl)
        n = a_repl + b_repl + 2                    # Wan total frames (incl. 2 pinned keyframes)
        if (n - 1) % 4 != 0:
            raise ValueError(
                f"generative frame count N={n} must be 4k+1; adjust a_keep/b_keep "
                f"(e.g. a_keep=2,b_keep=2 -> N=5). Got a_keep={a_keep}, b_keep={b_keep}."
            )

        # KEPT real keyframes straddling the seam; the strict middle is regenerated.
        kf_first = window.anchor[-a_keep]          # last KEPT A frame
        kf_last = window.b_original[b_keep]        # first KEPT B frame

        if gcfg.engine == "vace":
            from vbf.generate.vace import VaceGenerator

            vg = VaceGenerator(model_id=gcfg.vace_model_id, dtype=gcfg.dtype,
                               cpu_offload=gcfg.cpu_offload, flow_shift=gcfg.flow_shift,
                               device=self.device, quantize=gcfg.quantize)
            frames = vg.flf2v(
                kf_first, kf_last, num_frames=n, height=gcfg.vace_height, width=gcfg.vace_width,
                steps=gcfg.num_inference_steps, guidance=gcfg.guidance_scale, prompt=gcfg.prompt, seed=gcfg.seed,
            ).to(self.device)
        else:  # wan_i2v
            pipe = self._pipeline()
            gen = torch.Generator(device="cpu").manual_seed(gcfg.seed)
            result = pipe(
                image=_to_pil(kf_first, size_wh), last_image=_to_pil(kf_last, size_wh),
                prompt=gcfg.prompt, height=gcfg.height, width=gcfg.width, num_frames=n,
                num_inference_steps=gcfg.num_inference_steps, guidance_scale=gcfg.guidance_scale,
                generator=gen, output_type="pil",
            )
            frames = _frames_to_tensor(result.frames[0]).to(self.device)  # [N,3,H,W] gen res
        middle = frames[1:-1]                       # [a_repl + b_repl, 3, H, W]

        if gcfg.domain_match:
            real_ref = torch.cat([window.anchor[-4:], window.b_original[:4]], 0).to(self.device)
            real_ref = F.interpolate(real_ref, size=middle.shape[-2:], mode="bilinear", align_corners=False)
            middle = _domain_match(middle, real_ref)
        elif gcfg.color_match:
            middle = _color_match(middle, torch.stack([kf_first, kf_last], 0))

        if self.logger is not None:
            strip = F.interpolate(frames, size=window.anchor.shape[-2:], mode="bilinear", align_corners=False)
            snap = BoundaryWindow(
                anchor=window.anchor[-1:], b=strip,
                b_original=window.b_original, optimize_count=strip.shape[0],
            )
            self.logger.log_step(
                step=0, losses={"loss/total": 0.0},
                metrics={"gen/num_frames": float(n), "gen/replaced": float(middle.shape[0])},
                window=snap,
            )

        return window, BridgeReplaceCorrection(middle.detach().cpu(), a_repl, b_repl)
