"""BoundaryRefiner — the single extension point of the framework.

Every boundary-improvement *method* implements ``refine``. The prototype is a
gradient optimizer; future methods (generative inpainting, online learning, RL,
ensembles) implement the same interface and reuse the same metrics/loss, so
swapping methods is a config change (``refiner.backend``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from vbf.config import Config
from vbf.data.window import BoundaryWindow
from vbf.losses.boundary_loss import BoundaryLoss

REFINERS: dict[str, type["BoundaryRefiner"]] = {}


def register_refiner(name: str) -> Callable[[type["BoundaryRefiner"]], type["BoundaryRefiner"]]:
    def deco(cls: type["BoundaryRefiner"]) -> type["BoundaryRefiner"]:
        if name in REFINERS:
            raise ValueError(f"refiner '{name}' already registered")
        cls.backend = name
        REFINERS[name] = cls
        return cls

    return deco


def build_refiner(config: Config, device: torch.device | str, logger=None) -> "BoundaryRefiner":
    name = config.refiner.backend
    if name not in REFINERS:
        raise KeyError(f"unknown refiner '{name}'. registered: {sorted(REFINERS)}")
    return REFINERS[name](config, device, logger)


class Correction(ABC):
    """A resolution-independent correction the refiner learned for B's frames.

    Decouples *optimizing* (done at working resolution) from *applying* (done at
    native resolution when writing the final video). This is what prevents a
    geometric warp from being re-introduced as additive ghosting at full res:
    each backend applies its own correction type natively (warp resamples the
    sharp native frame; gradient adds an upsampled delta).
    """

    optimize_count: int = 0
    a_tail_count: int = 0   # how many of A's LAST frames this correction also modifies (0 = none)

    @abstractmethod
    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        """Apply the correction to native B frame ``i`` ([1,3,H,W] float in [0,1])."""

    def apply_a_frame(self, frame_native: torch.Tensor, j: int) -> torch.Tensor:
        """Apply correction to A's tail frame ``j`` (0 = first of the last ``a_tail_count``).

        Default no-op; backends that ramp the correction onto A's tail override this.
        """
        return frame_native


class ColorCorrection(Correction):
    """Per-frame per-channel color/exposure transform: ``gain * frame + bias``."""

    def __init__(self, gain: torch.Tensor, bias: torch.Tensor):
        self.gain = gain  # [M, 3]
        self.bias = bias  # [M, 3]
        self.optimize_count = gain.shape[0]

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        g = self.gain[i].to(frame_native.device).view(1, 3, 1, 1)
        b = self.bias[i].to(frame_native.device).view(1, 3, 1, 1)
        return (frame_native * g + b).clamp(0, 1)


class PassThroughCorrection(Correction):
    """No-op correction (returns frames unchanged). Safety fallback when a learned
    correction would make the seam WORSE than the raw clips."""

    def __init__(self, optimize_count: int = 0):
        self.optimize_count = optimize_count

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        return frame_native


class ReplaceCorrection(Correction):
    """Replaces B's first M frames with generated/synthesized frames.

    Used by generative backends that *synthesize* transition content rather than
    transform the original frame. Holds frames ``[M, 3, H, W]`` (float in [0,1]);
    they are resized to the target frame size on apply.
    """

    def __init__(self, frames: torch.Tensor):
        self.frames = frames  # [M, 3, H, W]
        self.optimize_count = frames.shape[0]

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        import torch.nn.functional as F

        gen = self.frames[i : i + 1].to(frame_native.device)
        if gen.shape[-2:] != frame_native.shape[-2:]:
            gen = F.interpolate(gen, size=frame_native.shape[-2:], mode="bilinear", align_corners=False)
        return gen.clamp(0, 1)


class BridgeReplaceCorrection(Correction):
    """Replaces a window straddling the seam with the SAME number of generated frames.

    Removes A's last ``a_frames`` and B's first ``b_frames`` and substitutes
    ``frames`` (count == a_frames + b_frames), so total length is unchanged —
    essential for timing-sensitive content (e.g. dance). The generated endpoints
    are conditioned on the window's edge frames, so they blend with the kept
    neighbors and carry the local motion speed across the seam.
    """

    def __init__(self, frames: torch.Tensor, a_frames: int, b_frames: int):
        self.frames = frames        # [a_frames + b_frames, 3, H, W]
        self.a_frames = a_frames
        self.b_frames = b_frames
        self.optimize_count = 0

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        return frame_native


class InsertCorrection(Correction):
    """Inserts synthesized bridge frames BETWEEN A and B (keeps A and B intact).

    Used by generative inbetweening: the bridge's endpoints are conditioned on
    A's last frame and B's first frame, so inserting the middle frames connects
    the clips without altering either or creating a new seam. ``apply_frame`` is
    a no-op (B is unchanged); the pipeline reads ``frames`` directly.
    """

    def __init__(self, frames: torch.Tensor):
        self.frames = frames  # [K, 3, H, W] inserted between A and B
        self.optimize_count = 0

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        return frame_native


class CompositeCorrection(Correction):
    """Chains corrections in order (e.g. geometric warp, then color/exposure)."""

    def __init__(self, corrections: list[Correction]):
        self.corrections = corrections
        self.optimize_count = max((c.optimize_count for c in corrections), default=0)

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        for c in self.corrections:
            if i < c.optimize_count:
                frame_native = c.apply_frame(frame_native, i)
        return frame_native


class BoundaryRefiner(ABC):
    backend: str = "base"

    def __init__(self, config: Config, device: torch.device | str, logger=None):
        self.config = config
        self.device = device
        self.logger = logger
        self.loss_fn = BoundaryLoss(config, device)

    @abstractmethod
    def refine(self, window: BoundaryWindow) -> tuple[BoundaryWindow, Correction]:
        """Refine B's frames to smooth the seam.

        Returns the (working-res) refined window for logging/metrics, plus a
        ``Correction`` that the pipeline applies to the native-res B frames.
        Implementations should call ``self.logger.log_step(...)`` per step if a
        logger is attached.
        """
