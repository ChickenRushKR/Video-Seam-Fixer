"""Boundary debug visualization: per boundary, a labeled panel
[prev_last | next_raw | raw_diff x5 | next_corrected | corrected_diff x5]
plus the mean|diff| before/after. Saves to analysis/debug/."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from vbf.io.video_io import load_video, to_float

OUT = "analysis/debug"
os.makedirs(OUT, exist_ok=True)
PH = 420  # panel height for the composite

# (name, prev_video, next_video, corrected_result, corrected_next0_idx)
BOUNDS = [
    ("v2_AB", "samples/video2-A.mp4", "samples/video2-B.mp4", "experiments/v2c_AB/result.mp4", 241),
    ("v2_BC", "samples/video2-B.mp4", "samples/video2-C.mp4", "experiments/v2c_BC/result.mp4", 289),
    ("v2_CD", "samples/video2-C.mp4", "samples/video2-D.mp4", "experiments/v2c_CD/result.mp4", 145),
    ("v3_AB", "samples/video3-A.mp4", "samples/video3-B.mp4", "experiments/v3c_AB/result.mp4", 241),
    ("v3_BC", "samples/video3-B.mp4", "samples/video3-C.mp4", "experiments/v3c_BC/result.mp4", 289),
    ("v3_CD", "samples/video3-C.mp4", "samples/video3-D.mp4", "experiments/v3c_CD/result.mp4", 145),
]


def to_img(t):  # [1,3,H,W] -> PIL, scaled to height PH
    a = (t[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((int(im.width * PH / im.height), PH))


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((3, 3), text, fill=(255, 255, 0))
    return im


for name, pv, nv, corr, idx in BOUNDS:
    prev = to_float(load_video(pv).frames[-1:])
    nxt_raw = to_float(load_video(nv).frames[:1])
    nxt_cor = to_float(load_video(corr).frames[idx : idx + 1])
    draw5 = lambda t: (t - prev).abs().clamp(0, 1) * 5
    raw_d, cor_d = draw5(nxt_raw), draw5(nxt_cor)
    rm, cm = float((nxt_raw - prev).abs().mean()), float((nxt_cor - prev).abs().mean())
    panels = [
        label(to_img(prev), "prev last"),
        label(to_img(nxt_raw), "next RAW"),
        label(to_img(raw_d), "RAW diff x5  %.4f" % rm),
        label(to_img(nxt_cor), "next FIXED"),
        label(to_img(cor_d), "FIXED diff x5  %.4f" % cm),
    ]
    W = sum(p.width for p in panels) + 4 * (len(panels) - 1)
    comp = Image.new("RGB", (W, PH), (20, 20, 20))
    x = 0
    for p in panels:
        comp.paste(p, (x, 0))
        x += p.width + 4
    comp.save(f"{OUT}/{name}.png")
    print(f"{name}: raw {rm:.4f} -> fixed {cm:.4f}  ({100*(1-cm/rm):.0f}% better)  -> {OUT}/{name}.png")
