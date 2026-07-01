"""CLI wrapper for the v1 chain boundary-normalizer (vbf.normalize.chain).

    python scripts/chain_normalize.py A.mp4 B.mp4 C.mp4 D.mp4 --out experiments/FINAL/run --mode tight

Back-compat: `python scripts/chain_normalize.py 3` runs the video3 sample chain.
"""

import argparse

from vbf.normalize import normalize_chain


def main():
    ap = argparse.ArgumentParser(description="Chain boundary normalization (v1)")
    ap.add_argument("clips", nargs="+", help="clip paths in order (or a single '2'/'3' sample id)")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--mode", default="tight", choices=["tight", "balanced"])
    ap.add_argument("--no-drop-dup", action="store_true", help="keep the duplicate seam frame")
    ap.add_argument("--no-slow", action="store_true", help="skip the boundary slow-mo comparison")
    args = ap.parse_args()

    clips = args.clips
    if len(clips) == 1 and clips[0] in ("2", "3"):        # sample shortcut
        v = clips[0]
        clips = [f"samples/video{v}-{c}.mp4" for c in "ABCD"]
        out = args.out or f"experiments/FINAL/video{v}_v1_{args.mode}"
    else:
        out = args.out or "experiments/FINAL/chain_run"

    def prog(stage, frac, msg):
        print(f"[{frac*100:5.1f}%] {stage:8} {msg}")

    r = normalize_chain(clips, out, mode=args.mode, drop_dup=not args.no_drop_dup,
                        make_slow=not args.no_slow, progress=prog)
    print(f"\n{r.num_frames}f @ {r.fps}fps in {r.seconds}s -> {r.full_path}")
    if r.slow_path:
        print(f"boundary slow-mo -> {r.slow_path}")
    print("scale x/y %:", r.transforms["scale_xy_pct"])
    print("colour gain:", r.transforms["colour_gain"], " sharpness:", r.transforms["sharpness"],
          " lighting:", r.transforms["lighting_gain"])
    for s in r.seams:
        print(f"  seam {s['pair']}: raw_gap {s['raw_gap']}  scale {s['scale_x_pct']}/{s['scale_y_pct']}%")


if __name__ == "__main__":
    main()
