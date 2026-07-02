"""Video <-> tensor I/O (memory-frugal).

Decoded frames are kept as **uint8** ``[T, 3, H, W]`` (RGB) — a native
1080x1920 clip is ~1.8 GB as uint8 vs ~7 GB as float32. Conversion to float in
[0, 1] and resizing happen in chunks so the full-resolution float tensor is
never materialized. Output is written frame-by-frame via a streaming writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Video:
    """A decoded video: frames ``[T, 3, H, W]`` **uint8** (RGB), plus fps."""

    frames: torch.Tensor  # uint8
    fps: float

    @property
    def num_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        return (self.frames.shape[2], self.frames.shape[3])  # (H, W)


def load_video(path: str | Path) -> Video:
    """Load a video into a ``Video`` with uint8 frames ``[T, 3, H, W]``."""
    path = str(path)
    meta = iio.immeta(path, plugin="pyav")
    fps = float(meta.get("fps", 24.0))

    arr = iio.imread(path, plugin="pyav")  # [T, H, W, 3] uint8
    if arr.ndim == 3:
        arr = arr[None, ...]
    frames = torch.from_numpy(np.ascontiguousarray(arr))  # uint8
    frames = frames.permute(0, 3, 1, 2).contiguous()       # [T, 3, H, W]
    return Video(frames=frames, fps=fps)


def to_float(frames_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [.,3,H,W] -> float in [0,1]."""
    return frames_u8.float() / 255.0


def to_uint8(frames_f: torch.Tensor) -> torch.Tensor:
    """float [.,3,H,W] in [0,1] -> uint8 (clamped, rounded)."""
    return (frames_f.detach().clamp(0, 1) * 255.0).round().to(torch.uint8)


def resize_frames(frames_f: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    """Bilinearly resize float frames ``[T,3,H,W]`` to ``size`` (H, W). No-op if None."""
    if size is None:
        return frames_f
    h, w = size
    if (frames_f.shape[2], frames_f.shape[3]) == (h, w):
        return frames_f
    return F.interpolate(frames_f, size=(h, w), mode="bilinear", align_corners=False)


def to_float_resized(
    frames_u8: torch.Tensor, size: tuple[int, int] | None, chunk: int = 32
) -> torch.Tensor:
    """Convert uint8 frames to float and resize, in chunks (avoids a full-res float copy)."""
    out = []
    for i in range(0, frames_u8.shape[0], chunk):
        block = to_float(frames_u8[i : i + chunk])
        out.append(resize_frames(block, size))
    return torch.cat(out, dim=0)


def save_video_stream(
    path: str | Path, frames_iter: Iterable[torch.Tensor], fps: float, crf: int = 16
) -> int:
    """Write an iterable of uint8 frames ``[3,H,W]`` to an mp4. Returns frame count.

    Encodes with an explicit x264 ``crf`` (default 16 = visually near-lossless); without it
    imageio falls back to a mid-quality default that visibly softens the output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    writer = iio2.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=1,  # don't silently resize dims up to a multiple of 16
        output_params=["-crf", str(crf), "-preset", "medium"],
    )
    try:
        for frame in frames_iter:
            arr = frame.detach().cpu().permute(1, 2, 0).contiguous().numpy()  # [H,W,3]
            writer.append_data(arr)
            n += 1
    finally:
        writer.close()
    return n


def save_video(path: str | Path, frames_f: torch.Tensor, fps: float) -> int:
    """Convenience: save a float tensor ``[T,3,H,W]`` in [0,1] (used by tests)."""
    def gen() -> Iterator[torch.Tensor]:
        for i in range(frames_f.shape[0]):
            yield to_uint8(frames_f[i])

    return save_video_stream(path, gen(), fps)
