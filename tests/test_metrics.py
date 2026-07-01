"""Metric sanity: ~0 on a perfectly continuous sequence, larger with a seam jump."""

import torch

from vbf.config import FlickerMetricConfig, RatioMetricConfig
from vbf.metrics.flicker import FlickerMetric
from vbf.metrics.ratio import RatioMetric


def _ramp(t: int, h: int = 16, w: int = 16) -> torch.Tensor:
    # smoothly brightening sequence: small constant frame-to-frame change
    base = torch.linspace(0.2, 0.5, t).view(t, 1, 1, 1)
    return base.expand(t, 3, h, w).contiguous()


def test_flicker_seam_increases_with_jump():
    m = FlickerMetric(FlickerMetricConfig(), "cpu")
    seq = _ramp(8)
    bi = 4
    clean = m.loss(seq, bi).item()

    jumped = seq.clone()
    jumped[bi:] += 0.4  # inject a colour jump at the seam
    dirty = m.loss(jumped, bi).item()

    assert dirty > clean
    assert m.observe(jumped, bi)["flicker/seam_diff"] > m.observe(seq, bi)["flicker/seam_diff"]


def test_ratio_seam_increases_with_exposure_shift():
    m = RatioMetric(RatioMetricConfig(), "cpu")
    seq = _ramp(8)
    bi = 4
    clean = m.loss(seq, bi).item()

    jumped = seq.clone()
    jumped[bi:] *= 1.8  # exposure shift at the seam
    dirty = m.loss(jumped, bi).item()

    assert dirty > clean
