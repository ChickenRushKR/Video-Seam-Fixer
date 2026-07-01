"""Boundary evaluation harness.

For each of the 6 boundaries (video2/3, AB/BC/CD) and each available condition
(raw + whatever corrected runs exist), compute a metric panel and dump it to
``analysis/eval/metrics.json`` plus a console table.

The panel separates "did the seam get closer" (seam L1, baseline-relative ratio)
from "is it perceptually/structurally still fine" (SSIM, Lab ΔE, temporal
instability) and "how much did we disturb B" (correction magnitude, downstream
integrity). See ``vbf/metrics/perceptual.py`` for definitions.

Conditions map a boundary tag to an experiment result dir; missing dirs are
skipped, so this runs before/after the conservative-mode runs exist.

Usage:
    python scripts/eval_boundaries.py
"""

import json
import os

import torch

from vbf.io.video_io import load_video, to_float
from vbf.metrics.perceptual import (
    delta_e76,
    motion_baseline,
    seam_l1,
    ssim,
    temporal_instability,
)

OUT = "analysis/eval"
os.makedirs(OUT, exist_ok=True)

# (tag, prev_clip, next_clip, next0_idx == prev clip length)
BOUNDS = [
    ("v2_AB", "samples/video2-A.mp4", "samples/video2-B.mp4", 241),
    ("v2_BC", "samples/video2-B.mp4", "samples/video2-C.mp4", 289),
    ("v2_CD", "samples/video2-C.mp4", "samples/video2-D.mp4", 145),
    ("v3_AB", "samples/video3-A.mp4", "samples/video3-B.mp4", 241),
    ("v3_BC", "samples/video3-B.mp4", "samples/video3-C.mp4", 289),
    ("v3_CD", "samples/video3-C.mp4", "samples/video3-D.mp4", 145),
]

# condition name -> run-dir template (tag "v2_AB" -> its result.mp4). raw is implicit.
# add rows here as new modes are produced (e.g. conservative -> "experiments/{run}cons/result.mp4").
def _run(tag: str, suffix: str) -> str:
    v, seam = tag.split("_")          # "v2", "AB"
    return f"experiments/{v}{suffix}_{seam}/result.mp4"

CONDITIONS = {
    "C3_full": lambda tag: _run(tag, "c"),          # existing aggressive cascade (v2c_AB ...)
    "C1_conservative": lambda tag: _run(tag, "cons"),
    "C2_cons_std": lambda tag: _run(tag, "consstd"),
    "C4_gated": lambda tag: _run(tag, "gate"),
}


def _corrected_next(path: str, idx: int):
    """Return (next0_frame_seq, full_corrected_B) from a stitched A+B result, or None."""
    if not os.path.exists(path):
        return None
    frames = to_float(load_video(path).frames)
    if idx >= frames.shape[0]:
        return None
    return frames[idx:]


def evaluate():
    report = {}
    for tag, pv, nv, idx in BOUNDS:
        prev = to_float(load_video(pv).frames)
        nxt = to_float(load_video(nv).frames)
        prev_last = prev[-1]
        base = motion_baseline(prev)

        raw = {
            "motion_baseline": base,
            "seam_l1": seam_l1(prev_last, nxt[0]),
            "seam_ratio": seam_l1(prev_last, nxt[0]) / base if base else float("nan"),
            "ssim": ssim(prev_last, nxt[0]),
            "deltaE": delta_e76(prev_last, nxt[0]),
            "temporal": temporal_instability(
                torch.cat([prev[-6:], nxt[:6]], dim=0), seam_index=6
            ),
        }
        conds = {"C0_raw": raw}

        for cname, pathfn in CONDITIONS.items():
            cor = _corrected_next(pathfn(tag), idx)
            if cor is None:
                continue
            n = min(cor.shape[0], nxt.shape[0])
            conds[cname] = {
                "seam_l1": seam_l1(prev_last, cor[0]),
                "seam_ratio": seam_l1(prev_last, cor[0]) / base if base else float("nan"),
                "ssim": ssim(prev_last, cor[0]),
                "deltaE": delta_e76(prev_last, cor[0]),
                "temporal": temporal_instability(
                    torch.cat([prev[-6:], cor[:6]], dim=0), seam_index=6
                ),
                "correction_magnitude": float((cor[:n] - nxt[:n]).abs().mean()),
                "downstream_integrity": float((cor[n - 1] - nxt[n - 1]).abs().mean()),
            }
        report[tag] = conds

    with open(f"{OUT}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _print_table(report)
    print(f"\n-> {OUT}/metrics.json")
    return report


def _print_table(report: dict):
    cols = ["seam_l1", "seam_ratio", "ssim", "deltaE", "temporal",
            "correction_magnitude", "downstream_integrity"]
    hdr = ["cond".ljust(18)] + [c[:9].rjust(10) for c in cols]
    for tag, conds in report.items():
        base = conds["C0_raw"]["motion_baseline"]
        print(f"\n=== {tag}   (motion baseline={base:.4f}) ===")
        print(" ".join(hdr))
        for cname, m in conds.items():
            row = [cname.ljust(18)]
            for c in cols:
                v = m.get(c)
                row.append(("%.4f" % v).rjust(10) if isinstance(v, float) else " " * 10)
            print(" ".join(row))


if __name__ == "__main__":
    evaluate()
