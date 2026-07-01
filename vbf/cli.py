"""Command-line entry point.

    vbf run --video-a A.mp4 --video-b B.mp4 [--config configs/default.yaml]
            [--steps N] [--run-id NAME] [--work HxW] [--optimize-b M]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from vbf.config import Config


def _parse_size(s: str | None):
    if not s:
        return None
    h, w = s.lower().split("x")
    return (int(h), int(w))


def _build_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config) if args.config else Config()
    if args.steps is not None:
        config.refiner.num_steps = args.steps
    if args.log_every is not None:
        config.refiner.log_every = args.log_every
    if args.lr is not None:
        config.refiner.lr = args.lr
    if args.backend is not None:
        config.refiner.backend = args.backend
    if args.warp_model is not None:
        config.refiner.warp.model = args.warp_model
    if args.gen_engine is not None:
        config.refiner.generative.engine = args.gen_engine
    if args.gen_steps is not None:
        config.refiner.generative.num_inference_steps = args.gen_steps
    if args.vace_model is not None:
        config.refiner.generative.vace_model_id = args.vace_model
    if args.vace_height is not None:
        config.refiner.generative.vace_height = args.vace_height
    if args.vace_width is not None:
        config.refiner.generative.vace_width = args.vace_width
    if args.flow_shift is not None:
        config.refiner.generative.flow_shift = args.flow_shift
    if args.vace_quantize is not None:
        config.refiner.generative.quantize = args.vace_quantize
    if args.domain_match:
        config.refiner.generative.domain_match = True
    if args.blend_frames is not None:
        config.refiner.cascade.blend_frames = args.blend_frames
    if args.no_scale_crop:
        config.refiner.cascade.scale_crop = False
    if args.motion_smooth:
        config.refiner.cascade.motion_smooth = True
    if args.color_histogram:
        config.refiner.cascade.color_histogram = True
    if args.scale_method is not None:
        config.refiner.cascade.scale_method = args.scale_method
    if args.structure_warp:
        config.refiner.cascade.structure_warp = True
    if args.bidirectional:
        config.refiner.cascade.bidirectional = True
    if args.structure_blur is not None:
        config.refiner.cascade.structure_blur = args.structure_blur
    if args.local_color:
        config.refiner.cascade.local_color = True
    if args.conservative:
        c = config.refiner.cascade
        c.conservative = True
        # least-destructive only: kill the components that manufacture artifacts
        c.structure_warp = False
        c.local_color = False
        c.color_histogram = False
        c.contrast = args.match_std   # mean-only colour by default (std overshoot -> tint)
        c.apply_decay = True          # fade the correction to 0 across the window
        c.max_scale = min(c.max_scale, 0.03)
        c.max_shift = min(c.max_shift, 0.02)
        c.blend_frames = 0
    if args.match_std and not args.conservative:
        config.refiner.cascade.contrast = True
    if args.strength_search:
        config.refiner.cascade.strength_search = True
    if args.gate_threshold is not None:
        config.refiner.cascade.gate_threshold = args.gate_threshold
    if args.gen_a_keep is not None:
        config.refiner.generative.a_keep = args.gen_a_keep
    if args.gen_b_keep is not None:
        config.refiner.generative.b_keep = args.gen_b_keep
    if args.work is not None:
        config.io.working_resolution = _parse_size(args.work)
    if args.output_res is not None:
        config.io.output_resolution = _parse_size(args.output_res)
    if args.optimize_b is not None:
        config.window.optimize_b_frames = args.optimize_b
    if args.experiments_dir is not None:
        config.logging.experiments_dir = args.experiments_dir
    return config


def _cmd_run(args: argparse.Namespace) -> None:
    from vbf.pipeline.stitcher import run

    config = _build_config(args)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    result = run(config, args.video_a, args.video_b, run_id=run_id)
    print(f"[vbf] run_id     : {run_id}")
    print(f"[vbf] run_dir    : {result.run_dir}")
    print(f"[vbf] result     : {result.result_path} ({result.num_frames}f @ {result.fps}fps)")
    print(f"[vbf] tensorboard: tensorboard --logdir {Path(result.run_dir) / 'tb'}")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate a canvas-aligned continuation clip conditioned on a previous frame."""
    import torch
    from vbf.generate.wan import WanGenerator
    from vbf.io.video_io import load_video, save_video, to_float

    if args.from_video:
        prev = load_video(args.from_video)
        cond = to_float(prev.frames[-1:])[0]  # last frame of previous clip
        fps = prev.fps
    else:
        raise SystemExit("provide --from-video (uses its last frame as the conditioning image)")
    last = None
    if args.to_video:
        nxt = load_video(args.to_video)
        last = to_float(nxt.frames[:1])[0]    # optional end keyframe (two-keyframe)

    gen = WanGenerator(model_id=args.model_id, dtype=args.dtype, cpu_offload=not args.no_offload)
    frames = gen.continue_from(
        cond, num_frames=args.num_frames, height=args.height, width=args.width,
        steps=args.steps, guidance=args.guidance, prompt=args.prompt, seed=args.seed, last_frame=last,
    )
    out = Path(args.out)
    n = save_video(out, frames, fps=args.fps or fps)
    print(f"[vbf] generated {n} frames -> {out} (cond = last frame of {args.from_video})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vbf", description="Video Boundary Fixer")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="optimize the boundary between two videos")
    run_p.add_argument("--video-a", required=True, help="video A (comes first)")
    run_p.add_argument("--video-b", required=True, help="video B (conditioned on A's last frame)")
    run_p.add_argument("--config", default="configs/default.yaml")
    run_p.add_argument("--steps", type=int, default=None)
    run_p.add_argument("--log-every", type=int, default=None)
    run_p.add_argument("--lr", type=float, default=None)
    run_p.add_argument("--backend", default=None, help="refiner backend: gradient|warp|warpcolor")
    run_p.add_argument(
        "--warp-model", default=None, choices=["translation", "affine", "polynomial"],
        help="geometric warp model (default: affine)",
    )
    run_p.add_argument("--gen-steps", type=int, default=None, help="generative diffusion inference steps")
    run_p.add_argument("--vace-model", default=None, help="VACE model id (e.g. Wan-AI/Wan2.1-VACE-14B-diffusers)")
    run_p.add_argument("--vace-height", type=int, default=None, help="VACE gen height (720p portrait: 1280)")
    run_p.add_argument("--vace-width", type=int, default=None, help="VACE gen width (720p portrait: 720)")
    run_p.add_argument("--flow-shift", type=float, default=None, help="UniPC flow_shift (5.0 for 720p, 3.0 for 480p)")
    run_p.add_argument("--vace-quantize", default=None, choices=["none", "int8wo", "float8wo", "float8dq"],
                       help="torchao quantization to fit 14B in VRAM (int8wo recommended)")
    run_p.add_argument("--domain-match", action="store_true",
                       help="histogram(color)+grain(noise) match generated frames to real neighbors")
    run_p.add_argument("--blend-frames", type=int, default=None, help="cascade transition blend frames (0 disables; avoids afterimage)")
    run_p.add_argument("--no-scale-crop", action="store_true", help="cascade: disable scale/crop step")
    run_p.add_argument("--motion-smooth", action="store_true", help="cascade: enable optical-flow motion smoothing")
    run_p.add_argument("--color-histogram", action="store_true", help="cascade: per-channel histogram color match (finer)")
    run_p.add_argument("--scale-method", default=None, choices=["flow", "feature"], help="cascade scale/crop estimator")
    run_p.add_argument("--structure-warp", action="store_true", help="cascade: local smoothed flow warp B->A structure")
    run_p.add_argument("--bidirectional", action="store_true", help="cascade: ramp warp over BOTH A tail and B head (frame-by-frame)")
    run_p.add_argument("--structure-blur", type=int, default=None, help="cascade flow low-pass kernel (smaller=finer; subject mask prevents jelly)")
    run_p.add_argument("--local-color", action="store_true", help="cascade: spatially-varying lighting/color match (background)")
    run_p.add_argument("--conservative", action="store_true", help="cascade: weak/local/faded adaptation (mean color + weak scale + grain; structure/local/LUT off, decay on)")
    run_p.add_argument("--match-std", action="store_true", help="cascade: also match per-channel std/contrast (default off in conservative)")
    run_p.add_argument("--strength-search", action="store_true", help="cascade: scale correction to the min strength that reaches the motion baseline (0=passthrough)")
    run_p.add_argument("--gate-threshold", type=float, default=None, help="cascade: motion-baseline multiplier for strength search (default 1.0)")
    run_p.add_argument("--gen-engine", default=None, choices=["wan_i2v", "vace"],
                       help="generative engine: wan_i2v (Wan2.2 first-last) | vace (Wan-VACE flf2v)")
    run_p.add_argument(
        "--gen-a-keep", type=int, default=None,
        help="generative: keep keyframe at A[-a] (replaces A's last a-1 frames). a_keep+b_keep-1 must be 4k+1",
    )
    run_p.add_argument(
        "--gen-b-keep", type=int, default=None,
        help="generative: keep keyframe at B[b] (replaces B's first b frames)",
    )
    run_p.add_argument("--work", default=None, help="working resolution HxW, e.g. 540x960")
    run_p.add_argument("--output-res", default=None, help="output resolution HxW (downscale A/B to match gen res for fair test)")
    run_p.add_argument("--optimize-b", type=int, default=None, help="optimize first M frames of B")
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--experiments-dir", default=None)
    run_p.set_defaults(func=_cmd_run)

    gen_p = sub.add_parser("generate", help="generate a canvas-aligned continuation clip (Wan2.2 I2V)")
    gen_p.add_argument("--from-video", required=True, help="previous clip; its LAST frame conditions generation")
    gen_p.add_argument("--to-video", default=None, help="optional next clip; its FIRST frame as end keyframe")
    gen_p.add_argument("--out", required=True, help="output mp4 path")
    gen_p.add_argument("--num-frames", type=int, default=49, help="frames to generate (4k+1)")
    gen_p.add_argument("--height", type=int, default=1280)
    gen_p.add_argument("--width", type=int, default=720)
    gen_p.add_argument("--steps", type=int, default=30)
    gen_p.add_argument("--guidance", type=float, default=5.0)
    gen_p.add_argument("--prompt", default="smooth natural continuation of the same scene")
    gen_p.add_argument("--seed", type=int, default=0)
    gen_p.add_argument("--fps", type=float, default=None)
    gen_p.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    gen_p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    gen_p.add_argument("--no-offload", action="store_true", help="disable model CPU offload")
    gen_p.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
