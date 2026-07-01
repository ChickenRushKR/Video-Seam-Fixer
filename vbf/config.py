"""Configuration schema (pydantic) and YAML loader.

The whole framework is driven by a single ``Config`` object so that swapping a
refiner backend, metric weights, or resolution is a config change rather than a
code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class IOConfig(BaseModel):
    fps: Optional[float] = None
    # Optimization resolution. Prefer working_scale (preserves aspect ratio);
    # working_resolution is an explicit (H, W) override (can distort if it does
    # not match the source aspect ratio).
    working_scale: Optional[float] = None                 # e.g. 0.5 => half of native, aspect kept
    working_resolution: Optional[tuple[int, int]] = None  # (H, W) explicit override
    output_resolution: Optional[tuple[int, int]] = None   # None => native source size


class WindowConfig(BaseModel):
    anchor_frames: int = 8           # K: last K frames of A held fixed
    optimize_b_frames: Optional[int] = None  # M: optimize first M frames of B (None => all)


class WarpConfig(BaseModel):
    """Flow-target geometric-warp refiner.

    Warps the *original sharp* B frames toward a FIXED target — a global warp fit
    to the excess seam flow ``fs - vA`` (estimated once), decaying to zero across
    B. Fixed target => stable (no RAFT feedback loop); resampling sharp frames =>
    no ghosting; a low-DOF GLOBAL warp => rigid (no per-pixel background wobble).
    """

    model: Literal["affine", "translation"] = "affine"  # global warp DOF (rigid, no jelly)
    strength: float = 1.0    # fraction of the converged correction to apply
    max_disp: float = 0.1    # per-iteration clamp on the excess field (guards bad flow)
    fit_size: int = 32       # downsample flow to this grid for a robust global fit
    plate_frames: int = 8    # frames averaged (temporal median) into each background plate


class ColorConfig(BaseModel):
    """Per-channel color/exposure matching of B's seam frame to A's last frame."""

    strength: float = 1.0       # fraction of the A<-B color difference to correct
    match_std: bool = True      # also match contrast (per-channel std), not just mean
    gain_clamp: float = 2.0     # clamp gain to [1/clamp, clamp] (guards flat channels)


class GenerativeConfig(BaseModel):
    """Diffusion video transition synthesis (Branch 1).

    Default uses the cached Wan2.2-TI2V-5B via WanImageToVideoPipeline, which
    supports TWO-keyframe conditioning (``image`` = A's last frame, ``last_image``
    = B boundary frame) — a content-aware morph between the clips, not a blend.
    """

    engine: Literal["wan_i2v", "vace"] = "wan_i2v"  # wan_i2v=Wan2.2 first-last; vace=Wan-VACE flf2v
    pipeline_class: str = "WanImageToVideoPipeline"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    vace_model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vace_height: int = 832   # VACE 1.3B is 480p (portrait HxW = 832x480)
    vace_width: int = 480
    flow_shift: float = 3.0  # UniPC flow_shift (3.0 for 480p, 5.0 for 720p)
    quantize: str = "none"   # none | int8wo | float8wo | float8dq (torchao); fits 14B in VRAM
    # Length-preserving bridge: keep real keyframes A[-a_keep] and B[b_keep], regenerate the
    # frames strictly between them. Replaced count = (a_keep-1)+b_keep; Wan frames N = that+2
    # (must be 4k+1). Defaults a_keep=2,b_keep=2 => replace 3 frames (A[-1],B[0],B[1]), N=5.
    a_keep: int = 2
    b_keep: int = 2
    color_match: bool = True                 # match generated frames' color/exposure to originals
    domain_match: bool = False               # stronger: histogram(color) + grain(noise) match to real neighbors
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    height: int = 1280                       # Wan 720p portrait (H>W), multiples of 16
    width: int = 720
    cpu_offload: bool = True                 # enable model CPU offload (fits 5B comfortably)
    prompt: str = "smooth natural continuation of the same scene, same subject and background"
    seed: int = 0


class CascadeConfig(BaseModel):
    """Least-destructive boundary cascade on REAL B frames (generation is last resort).

    Steps (each toggleable), escalating: color -> exposure/contrast -> sharpness/grain
    -> scale/crop -> (motion smoothing) -> short transition blend. Analysis compares
    A's last ~0.5-1s vs B's first ~0.5-1s; corrections fix B toward A, length preserved.
    """

    analysis_frames: int = 18     # ~0.75s @24fps window for A_tail / B_head difference analysis
    color: bool = True            # 1. per-channel mean match (color)
    contrast: bool = True         # 2. per-channel std match (exposure / contrast)
    grain: bool = True            # 3. match B's grain/noise level to A
    scale_crop: bool = True       # 4. global scale+shift align B to A (clamped, real-frame crop)
    scale_method: str = "flow"    #    flow (RAFT line-fit) | feature (ORB+RANSAC on static background)
    max_scale: float = 0.06       #    clamp scale correction to +/-6%
    max_shift: float = 0.04       #    clamp shift to +/-4% (normalized)
    motion_smooth: bool = False   # 5. optical-flow MOTION-COMPENSATED transition (A flow-aligned to B,
                                  #    then dissolve appearance) — smooths residual WITHOUT afterimage
    motion_frames: int = 5        #    motion-compensated transition length
    structure_warp: bool = False  # 4b. LOCAL flow warp of B toward A's structure (heavily smoothed,
                                  #     decaying) — nudges local proportion; no A overlay (no ghost)
    structure_frames: int = 8     #     frames warped (fades out)
    structure_strength: float = 0.9
    bidirectional: bool = False   #     spread the warp over BOTH A's tail and B's head (frame-by-frame
                                  #     ratio ramp meeting at a midpoint) — smoother, OK to nudge A too
    ramp_a_frames: int = 8        #     A tail frames in the ramp
    ramp_b_frames: int = 8        #     B head frames in the ramp
    structure_blur: int = 25      #     flow low-pass kernel (smaller = finer alignment; too small -> jelly)
    structure_max_disp: float = 0.09  # clamp (raised so horizontal isn't under-corrected)
    subject_mask: bool = True     #     warp only static background (mask out moving subject) -> no feet drag
    color_histogram: bool = False # finer color match (per-channel histogram/LUT vs mean/std)
    color_hist_strength: float = 0.8  # partial histogram (0=mean/std only, 1=full; partial avoids over-correct)
    local_color: bool = False     # spatially-varying lighting/color: replace B's low-freq (lighting) with
                                  # A's, keep B's detail; background only (subject masked). Needs structure aligned.
    local_color_blur: int = 61    # low-freq kernel for the lighting field (large = smooth illumination only)
    local_color_clamp: float = 1.6  # clamp per-region gain
    local_color_strength: float = 1.0
    blend_frames: int = 0         # 6. transition cross-dissolve (OFF by default — held-A_last blend
                                  #    causes an afterimage; only enable if a residual seam needs it)
    apply_decay: bool = False     # corrections constant over B (False) or fade out (True)
    # --- conservative adaptation mode ------------------------------------------------
    # Reframes the cascade as a WEAK, LOCAL, FADED adaptation of B's head to A's tail
    # rather than an aggressive full-B correction. Only touches the first
    # ``window.optimize_b_frames`` frames (tail untouched => no chain propagation to C),
    # ramps the correction to zero across that window (apply_decay), and uses only the
    # least-destructive components (mean colour + weak scale/crop + grain; structure warp,
    # local colour and histogram LUT are off). The preset is applied in cli.py.
    conservative: bool = False
    strength_search: bool = False  # scale the whole correction to the MINIMUM strength that
                                   # brings the seam down to the scene's motion baseline (0 => passthrough)
    gate_threshold: float = 1.0    # target = motion_baseline * gate_threshold (<1 = stricter)


class RefinerConfig(BaseModel):
    backend: str = "warpcolor"
    num_steps: int = 200
    log_every: int = 20
    lr: float = 0.02
    optimizer: Literal["adam", "sgd"] = "adam"
    # constant=True applies the estimated drift correction UNIFORMLY to every
    # optimized B frame (no decay). Right when the A->B mismatch is a constant
    # generation drift (subtle scale/colour), not a one-frame seam jump: keeps B
    # internally consistent instead of drifting back within the window.
    constant: bool = True
    warp: WarpConfig = Field(default_factory=WarpConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    generative: GenerativeConfig = Field(default_factory=GenerativeConfig)


class FlickerMetricConfig(BaseModel):
    color_space: Literal["rgb", "lab"] = "rgb"


class RatioMetricConfig(BaseModel):
    stats: list[Literal["mean", "std"]] = Field(default_factory=lambda: ["mean", "std"])


class FlowMetricConfig(BaseModel):
    model: str = "raft_large"
    warp_consistency: bool = True
    radius: int = 3                       # pairs within +/- radius of the seam
    num_flow_updates: int = 6             # RAFT refinement iters (12 default is heavier)
    flow_resolution: Optional[tuple[int, int]] = None  # (H,W) for RAFT estimation; None => working res


class MetricConfigs(BaseModel):
    flicker: FlickerMetricConfig = Field(default_factory=FlickerMetricConfig)
    ratio: RatioMetricConfig = Field(default_factory=RatioMetricConfig)
    flow: FlowMetricConfig = Field(default_factory=FlowMetricConfig)


class LossConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"flicker": 1.0, "ratio": 0.5, "flow": 1.0, "fidelity": 5.0}
    )
    metrics: MetricConfigs = Field(default_factory=MetricConfigs)


class LoggingConfig(BaseModel):
    tensorboard: bool = True
    jsonl: bool = True
    save_snapshots: bool = True
    experiments_dir: str = "experiments"


class Config(BaseModel):
    io: IOConfig = Field(default_factory=IOConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    refiner: RefinerConfig = Field(default_factory=RefinerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)
