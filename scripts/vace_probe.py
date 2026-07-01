"""Feasibility probe for Wan-VACE: load on this env (transformers 5.x) and run a
minimal first-last interpolation (keep endpoints, generate middle).

Builds a VACE inpainting-along-time task:
  video = [A_last, gray, ..., gray, B_first]   (length num_frames)
  mask  = [black,  white, ..., white, black ]   (black=keep, white=generate)
  reference_images = [A_last]                    (appearance anchor)
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vbf.io.video_io import load_video, to_float, save_video

MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
H, W = 832, 480          # Wan2.1-VACE-1.3B is a 480p model (portrait HxW)
NUM = 17                 # 4k+1
A_PATH = "samples/video2-A.mp4"
B_PATH = "samples/video2-B.mp4"
OUT = "experiments/gen/vace_firstlast.mp4"


def to_pil(frame):  # frame [3,h,w] float -> PIL (W,H)
    f = F.interpolate(frame.unsqueeze(0).float().clamp(0, 1), size=(H, W), mode="bilinear", align_corners=False)[0]
    return Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def main():
    a = load_video(A_PATH); b = load_video(B_PATH)
    a_last = to_pil(to_float(a.frames[-1:])[0])
    b_first = to_pil(to_float(b.frames[:1])[0])
    gray = Image.new("RGB", (W, H), (128, 128, 128))
    black = Image.new("L", (W, H), 0)
    white = Image.new("L", (W, H), 255)

    video = [a_last] + [gray] * (NUM - 2) + [b_first]
    mask = [black] + [white] * (NUM - 2) + [black]

    from diffusers import WanVACEPipeline
    print("loading WanVACEPipeline...", flush=True)
    pipe = WanVACEPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    import inspect
    print("call params:", [p for p in inspect.signature(pipe.__call__).parameters if p != "self"], flush=True)

    gen = torch.Generator(device="cpu").manual_seed(0)
    out = pipe(
        prompt="a person dancing, smooth natural motion, same scene",
        video=video, mask=mask, reference_images=[a_last],
        height=H, width=W, num_frames=NUM,
        num_inference_steps=25, guidance_scale=5.0, generator=gen, output_type="pil",
    )
    frames = out.frames[0]
    t = torch.stack([torch.from_numpy(np.asarray(f).astype(np.float32) / 255).permute(2, 0, 1) for f in frames])
    n = save_video(OUT, t, fps=24.0)
    print(f"VACE OK: generated {n} frames -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
