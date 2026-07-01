"""End-to-end pipeline: load A/B -> build window -> refine -> stitch -> save.

Optimization runs at the working resolution (``io.working_scale``) for tractable
RAFT + B optimization. The refiner returns a resolution-independent
``Correction`` that is applied to the native-resolution B frames at output time,
so source detail is preserved (and a geometric warp stays ghost-free at full res).

Memory: frames are held as uint8; the full-res float video is never
materialized — working-res floats are built in chunks and the output is written
frame-by-frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from vbf.config import Config
from vbf.data.window import BoundaryWindow
from vbf.io.video_io import (
    Video,
    load_video,
    resize_frames,
    save_video_stream,
    to_float,
    to_float_resized,
    to_uint8,
)
from vbf.logging.run_logger import RunLogger
from vbf.refiners.base import (
    BridgeReplaceCorrection,
    Correction,
    InsertCorrection,
    PassThroughCorrection,
    build_refiner,
)


@dataclass
class StitchResult:
    run_dir: Path
    result_path: Path
    fps: float
    num_frames: int


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _work_size(native_hw: tuple[int, int], io_cfg) -> tuple[int, int] | None:
    """Resolve the optimization resolution (H, W). Prefer aspect-preserving scale."""
    if io_cfg.working_resolution:
        return tuple(io_cfg.working_resolution)
    if io_cfg.working_scale:
        h, w = native_hw
        s = io_cfg.working_scale
        return (max(8, round(h * s)), max(8, round(w * s)))
    return None


def _stitched_frames(
    a: Video,
    b: Video,
    correction: Correction,
    out_size: tuple[int, int] | None,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Yield uint8 ``[3,H,W]`` frames: all of A, then B with the seam correction.

    The correction is applied to the *native* B frame (warp resamples the sharp
    native frame; gradient adds an upsampled delta), so the full-res output keeps
    whatever quality property the backend guarantees (e.g. no ghosting for warp).
    """
    m = correction.optimize_count
    target = out_size or b.size  # generated frames share B's native size when out_size is None

    if isinstance(correction, BridgeReplaceCorrection):
        # length-preserving: drop A's last a_frames and B's first b_frames, splice generated
        a_keep_n = a.num_frames - correction.a_frames
        for i in range(a_keep_n):
            yield to_uint8(resize_frames(to_float(a.frames[i : i + 1]), out_size)[0])
        for k in range(correction.frames.shape[0]):
            yield to_uint8(resize_frames(correction.frames[k : k + 1].to(device), target)[0])
        for i in range(correction.b_frames, b.num_frames):
            yield to_uint8(resize_frames(to_float(b.frames[i : i + 1]).to(device), out_size)[0])
        return

    insert = isinstance(correction, InsertCorrection)
    ac = getattr(correction, "a_tail_count", 0)
    a_start = a.num_frames - ac
    for i in range(a.num_frames):
        f = to_float(a.frames[i : i + 1]).to(device)
        if ac and i >= a_start:
            f = correction.apply_a_frame(f, i - a_start)
        yield to_uint8(resize_frames(f, out_size)[0])
    if insert:  # bridge frames between A and B; A and B kept intact (changes length)
        for k in range(correction.frames.shape[0]):
            yield to_uint8(resize_frames(correction.frames[k : k + 1].to(device), target)[0])
    for i in range(b.num_frames):
        f = to_float(b.frames[i : i + 1]).to(device)  # [1,3,Hn,Wn]
        if not insert and i < m:
            f = correction.apply_frame(f, i)
        yield to_uint8(resize_frames(f, out_size)[0])


def run(
    config: Config,
    video_a: str | Path,
    video_b: str | Path,
    run_id: str,
    device: torch.device | None = None,
) -> StitchResult:
    device = device or _select_device()
    a: Video = load_video(video_a)
    b: Video = load_video(video_b)

    fps = config.io.fps or a.fps
    if abs(a.fps - b.fps) > 1e-3:
        raise ValueError(f"fps mismatch: A={a.fps} B={b.fps}. Re-encode to a common fps first.")

    work = _work_size(b.size, config.io)
    a_work = to_float_resized(a.frames, work)
    b_work = to_float_resized(b.frames, work)

    window = BoundaryWindow.build(
        a_frames=a_work,
        b_frames=b_work,
        anchor_frames=config.window.anchor_frames,
        optimize_b_frames=config.window.optimize_b_frames,
    )
    del a_work  # working A only needed for anchor (already copied into window)

    with RunLogger(config, run_id) as logger:
        logger.log_text(
            "inputs",
            f"A={video_a} ({a.num_frames}f) | B={video_b} ({b.num_frames}f) | "
            f"fps={fps} | work={work} | M={window.optimize_count} | device={device}",
        )
        refiner = build_refiner(config, device, logger)
        window, correction = refiner.refine(window)

        # NATIVE-res safety: if the correction makes the seam worse than raw, pass through.
        if correction.optimize_count > 0 and not isinstance(correction, (InsertCorrection, BridgeReplaceCorrection)):
            a_last_n = to_float(a.frames[-1:]).to(device)
            b0_n = to_float(b.frames[:1]).to(device)
            raw_e = float((b0_n - a_last_n).abs().mean())
            fix_e = float((correction.apply_frame(b0_n.clone(), 0) - a_last_n).abs().mean())
            if fix_e > raw_e * 1.03:
                logger.log_text("safety", f"correction worsened seam ({raw_e:.4f}->{fix_e:.4f}); passthrough")
                correction = PassThroughCorrection(correction.optimize_count)

        out_size = tuple(config.io.output_resolution) if config.io.output_resolution else None
        n = save_video_stream(
            logger.result_path,
            _stitched_frames(a, b, correction, out_size, device),
            fps=fps,
        )
        return StitchResult(
            run_dir=logger.root,
            result_path=logger.result_path,
            fps=fps,
            num_frames=n,
        )
