"""Per-run logging: TensorBoard scalars/images + JSONL metrics + disk snapshots.

A run writes to ``<experiments_dir>/<run_id>/``::

    config.yaml      snapshot of the config used
    metrics.jsonl    one line per logged step (losses + metrics)
    tb/              TensorBoard event files
    snapshots/       boundary strip PNGs per logged step
    result.mp4       final stitched video (written by the pipeline)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from vbf.config import Config
from vbf.data.window import BoundaryWindow


class RunLogger:
    def __init__(self, config: Config, run_id: str):
        self.config = config
        self.run_id = run_id
        self.root = Path(config.logging.experiments_dir) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")

        self._jsonl = None
        if config.logging.jsonl:
            self._jsonl = open(self.root / "metrics.jsonl", "w", encoding="utf-8")

        self._tb = None
        if config.logging.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.root / "tb"))

        if config.logging.save_snapshots:
            (self.root / "snapshots").mkdir(exist_ok=True)

    @property
    def result_path(self) -> Path:
        return self.root / "result.mp4"

    def log_step(
        self,
        step: int,
        losses: dict[str, float],
        metrics: dict[str, float],
        window: BoundaryWindow | None = None,
    ) -> None:
        record = {"step": step, **losses, **metrics}
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(record) + "\n")
            self._jsonl.flush()
        if self._tb is not None:
            for k, v in {**losses, **metrics}.items():
                self._tb.add_scalar(k, v, step)
        if window is not None and self.config.logging.save_snapshots:
            self._log_boundary(step, window)

    def _log_boundary(self, step: int, window: BoundaryWindow) -> None:
        # tight view around the seam: last 2 anchor frames | first 2 B frames,
        # at native aspect ratio (padding separates frames; seam is the middle gap)
        strip = window.preview(radius=2).detach().cpu().clamp(0, 1)
        grid = make_grid(strip, nrow=strip.shape[0], padding=4, pad_value=1.0)
        if self._tb is not None:
            self._tb.add_image("boundary_seam", grid, step)
        if self.config.logging.save_snapshots:
            save_image(grid, self.root / "snapshots" / f"step_{step:05d}.png")

    def log_text(self, tag: str, text: str) -> None:
        if self._tb is not None:
            self._tb.add_text(tag, text, 0)

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
