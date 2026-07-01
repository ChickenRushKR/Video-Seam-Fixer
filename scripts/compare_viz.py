"""Per-boundary visual comparison of correction conditions.

For each boundary, a labeled row:
  [ prev_last | raw B0 | conservative B0 | full B0 | cons |Δ|x8 | full |Δ|x8 ]
where the two rightmost panels are correction-magnitude heatmaps (|corrected - raw|,
amplified) — showing WHERE and HOW MUCH each condition disturbs B. Conservative should
be near-black (barely touched); full should light up (heavily altered).

Saves analysis/eval/compare_<tag>.png.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from vbf.io.video_io import load_video, to_float

OUT = "analysis/eval"
os.makedirs(OUT, exist_ok=True)
PH = 380

BOUNDS = [
    ("v2_AB", "samples/video2-A.mp4", "samples/video2-B.mp4", 241),
    ("v2_BC", "samples/video2-B.mp4", "samples/video2-C.mp4", 289),
    ("v2_CD", "samples/video2-C.mp4", "samples/video2-D.mp4", 145),
    ("v3_AB", "samples/video3-A.mp4", "samples/video3-B.mp4", 241),
    ("v3_BC", "samples/video3-B.mp4", "samples/video3-C.mp4", 289),
    ("v3_CD", "samples/video3-C.mp4", "samples/video3-D.mp4", 145),
]


def _path(tag, suffix):
    v, seam = tag.split("_")
    return f"experiments/{v}{suffix}_{seam}/result.mp4"


def to_img(t):
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((int(im.width * PH / im.height), PH))


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((3, 3), text, fill=(255, 255, 0))
    return im


def corrected_b0(tag, suffix, idx):
    p = _path(tag, suffix)
    if not os.path.exists(p):
        return None
    return to_float(load_video(p).frames[idx:idx + 1])[0]


for tag, pv, nv, idx in BOUNDS:
    prev_last = to_float(load_video(pv).frames[-1:])[0]
    raw = to_float(load_video(nv).frames[:1])[0]
    cons = corrected_b0(tag, "cons", idx)
    full = corrected_b0(tag, "c", idx)  # existing aggressive cascade
    if cons is None or full is None:
        print(f"skip {tag}: missing cons/full")
        continue
    heat = lambda c: ((c - raw).abs().mean(0, keepdim=True).repeat(3, 1, 1)) * 8
    panels = [
        label(to_img(prev_last), "prev last"),
        label(to_img(raw), "raw B0"),
        label(to_img(cons), "conservative B0"),
        label(to_img(full), "full B0"),
        label(to_img(heat(cons)), "cons |d|x8"),
        label(to_img(heat(full)), "full |d|x8"),
    ]
    W = sum(p.width for p in panels) + 4 * (len(panels) - 1)
    comp = Image.new("RGB", (W, PH), (20, 20, 20))
    x = 0
    for p in panels:
        comp.paste(p, (x, 0))
        x += p.width + 4
    comp.save(f"{OUT}/compare_{tag}.png")
    print(f"{tag}: cons|d|={float((cons-raw).abs().mean()):.4f}  full|d|={float((full-raw).abs().mean()):.4f}  -> {OUT}/compare_{tag}.png")
