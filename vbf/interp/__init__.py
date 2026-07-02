"""Temporal frame interpolation (v2).

Synthesize in-between frames at a seam to smooth the residual subject-motion step
that chain-normalization (v1) leaves behind. Backends share one interface,
``Interpolator.interpolate(f0, f1, t) -> f_t`` on ``[3,H,W]`` float frames.

- ``flow`` (default): reuses the cached RAFT + backward-warp from ``vbf.metrics.flow``
  — no extra weights/deps, deterministic. Good for the small, localized seam motion.
- ``rife`` (opt-in): learned interpolator (better on occlusion / large motion); needs
  external weights. Falls back to ``flow`` when unavailable.
"""

from vbf.interp.flow_interp import FlowInterpolator


def get_interpolator(backend: str = "flow", device=None, **kw):
    backend = (backend or "flow").lower()
    if backend == "flow":
        return FlowInterpolator(device=device, **kw)
    if backend == "rife":
        try:
            from vbf.interp.rife_interp import RifeInterpolator
            return RifeInterpolator(device=device, **kw)
        except Exception as e:  # weights/lib missing -> safe fallback
            print(f"[interp] rife unavailable ({e}); falling back to flow")
            return FlowInterpolator(device=device, **kw)
    raise ValueError(f"unknown interpolation backend: {backend}")


__all__ = ["get_interpolator", "FlowInterpolator"]
