<!-- Language: [🇰🇷 한국어](README.md) · **🇺🇸 English** -->

# seam-fixer

**Make the cut between consecutively-generated video clips invisible.**

When you extend a video with a generative model — clip **B** conditioned on the last frame of
clip **A**, then **C** on B, then **D** on C — concatenating them leaves a visible seam at every
cut. `seam-fixer` removes that seam with a fast, deterministic, post-hoc pass (no model, no
per-clip retraining) and ships both a CLI and a local web app.

```
A ──┐        ┌── B ──┐        ┌── C ──┐        ┌── D
    │  seam  │       │  seam  │       │  seam  │
    ▼        ▼       ▼        ▼       ▼        ▼
 [ scale/colour/exposure/sharpness/lighting drift + a duplicate freeze frame ]
                          │  seam-fixer
                          ▼
A ───────────────────────────────────────────────── D   (one continuous take)
```

---

## Representative use case (why this tool exists)

**Setup.** You want a long video from an image→video model but can't generate it all at once, so
you **extend clip by clip** — e.g. a person dancing by a river (portrait 1080×1920). Generate
clip A, then generate B **conditioned on A's last frame**, then C from B's end, then D from C's.
Concatenate the four into one piece.

**What you see.** On playback each cut **pops slightly bigger (zoom)**, the **brightness/colour
flickers**, and the motion **hitches** for a frame. Each is subtle but clearly visible at the
cut, and it **accumulates** — by D the image is noticeably more zoomed and brighter than A. It
"looks stitched."

![Seam example](docs/seam_example.png)

> Top: `A_last` (prev clip end) and `B_first` (next clip start) — nearly identical. Bottom-left:
> amplify their difference (×5) and **the whole frame's edges glow** — bridge, ground, lights =
> the seam (scale/position/colour mismatch, mean diff 0.032). Bottom-right: after alignment the
> background edges **go blue** (0.013); the orange that remains is **just the person** = genuine
> subject motion that alignment can't touch, which [Stage 2](#stage-2--seam-motion-interpolation-learned) handles.

**What had to be fixed (and the wrong turns).**
- First guessed it was *colour* → matching colour alone added a red cast and made it worse
  (measured: the colour difference was small).
- Then suspected *subject motion* (the diff map lit up the person) → but that was the pattern of
  a slight **global zoom** (every edge moves), not the subject.
- Measured properly: the real cause was **anisotropic scale (x≠y) + accumulating
  colour/exposure/sharpness/lighting drift + a duplicate freeze frame** — several overlapping
  issues, not one.
- And fixing one clip toward its predecessor changed that clip's *tail*, **breaking the next
  cut** (error propagation).

**So how it was built.** Align *all* clips into one shared reference (no propagation), then fill
the *residual subject motion* alignment can't remove with a **learned interpolator**. The
[two halves](#how-it-works-method--rationale) (deterministic alignment + learned interpolation)
are the conclusion of that process. Every decision was **verified with metrics**, not the eye
(colour ΔE, structural SSIM, each clip's own motion baseline, downstream integrity).

---

## The problem (what actually causes the seam)

Each generated clip is *almost* a continuation of the previous one, but not exactly. Measuring
the boundary between real generated clips shows the discontinuity is **not random** — it's a
small, consistent, per-clip drift that **accumulates down the chain**:

| what drifts | example (measured, clip A→D) | how you perceive it |
|---|---|---|
| **scale / ratio** (anisotropic!) | x **+4.9%**, y **+1.4%** | frame "pops" bigger at the cut |
| **colour / exposure** | each clip ~0.5–0.9% brighter | a flicker in tone |
| **sharpness** | each clip a bit sharper | texture "shimmers" |
| **lighting field** | spatially-varying (e.g. tunnel) | a lighting "wipe" |
| **duplicate frame** | B[0] ≈ A[-1] (the conditioning frame) | a 1-frame **freeze / stutter** |

Two findings drove the design:

1. **The scale drift is anisotropic** — horizontal and vertical scale differently, so a uniform
   (similarity) transform can't fix it; a **full affine** is required.
2. **The seam step is *smaller* than normal motion**, not larger — because B[0] duplicates A[-1].
   So the cut reads as a *hitch*, and dropping that duplicate frame restores continuous motion.

(An earlier per-clip "cascade" that corrected each clip independently made things *worse*: it
shifted a clip's whole body, breaking the *next* seam it fed into. See
[Design history](#design-history).)

---

## How it works (method + rationale)

The design has **two complementary halves**:

- **Stage 1 · Alignment (deterministic):** undo the *global* per-clip drift (scale, colour,
  exposure, sharpness, lighting) exactly, with no model.
- **Stage 2 · Interpolation (learned):** smooth the *residual subject motion* that alignment
  can't remove, by synthesizing in-between frames (RIFE).

Deterministic alignment is precise and stable for the global drift; a learned model is strong
where alignment can't reach (the temporal motion). Each tool is placed where it wins.

### Stage 1 — chain alignment (deterministic)

Treat the whole sequence as one chain and map **every clip into the first clip's reference
space**, so both sides of every cut end up in the same space → the seam matches and no
downstream seam is broken. Per clip, composed cumulatively to the reference:

1. **Geometry — full affine, background-locked.**
   Estimate a 6-DOF affine (independent x/y scale + shear + rotation + translation) from
   `next[0]` to `prev[-1]` with ORB + RANSAC. Crucially, features are matched on the **static
   background only** (a cheap motion mask removes the moving subject) — otherwise the subject's
   motion biases the global fit. Anisotropic scale is the reason a full affine (not a similarity
   transform) is used. Composed across seams; a centre crop-zoom removes any border.

2. **Colour + exposure.** Per-channel `gain·x + bias` (matching both mean and std → colour
   balance *and* exposure/contrast).

3. **Sharpness.** Match high-frequency energy (blur / unsharp) so one clip isn't crisper than
   its neighbour.

4. **Lighting.** A **spatially-varying low-frequency gain map** (small 24×14 grid = very smooth,
   can't form a blob), **subject-masked** so the person is never relit and gently clamped. This
   is what fixes location-dependent lighting (e.g. a bright-centre tunnel) that a global gain
   can't. Self-adaptive: ≈1 where lighting already matches, corrective where it differs.

5. **Drop the duplicate frame.** Each next clip's first frame ≈ the previous clip's last frame,
   so it's skipped — turning the seam freeze into a normal motion step.

**Why chain-consistent instead of "fix each seam locally"?** A local fix that moves a whole clip
toward its predecessor changes that clip's *tail*, which was the anchor the *next* clip was
generated from — so it just pushes the error downstream. Mapping everything to one reference
makes every seam consistent at once. Verified: correcting all four clips this way keeps
`downstream_integrity ≈ 0` (a clip's tail is not disturbed relative to what feeds off it).

**Two dead ends worth not repeating** (measured, not guessed):
- *ECC intensity refinement* of the affine **diverged** — global photometric alignment gets
  pulled off by the moving subject + lighting. Background-masked ORB is the clean path.
- *Cross-dissolving* the cut **ghosts** the moving subject (double exposure). The seam residual
  is genuine subject motion; blending is the wrong tool.

**Modes:** **`tight`** (default) — match each cut as closely as possible; correction is chained
to the first clip, so later clips carry more of it. **`balanced`** — map every clip to the
*average* look instead; each clip changes less (stays natural) at the cost of slightly looser
seam matching.

### Stage 2 — seam motion interpolation (learned)

After alignment, each seam's residual is the **subject-motion floor** — `prev[-1]` and `next[0]`
are different *moments*, so a subject in motion can't be aligned away. So we **synthesize K
in-between frames and insert them** at each seam (K=1 + duplicate-drop = length-preserving: the
dropped freeze is replaced by a real midpoint).

- **`flow`** (no deps): bidirectional RAFT flow warps pixels toward the midpoint + an
  occlusion-aware blend. Fine for small motion, but **fast/occluded parts (e.g. a swinging hand)
  tear or go translucent** — the fundamental limit of "moving pixels" (raising flow resolution
  doesn't help).
- **`rife`** (recommended): a learned network (RIFE v4.26, weights auto-downloaded via `ccvfi`)
  estimates intermediate flow + *hallucinates* occluded content + learns the fusion, so it
  **reconstructs the hand the flow backend tears** into a solid mid-pose. Arbitrary timestep, so
  K=2/3 work too. Falls back to `flow` if `ccvfi` is absent.

Because Stage 1 already aligned the two seam frames, the interpolator's input is clean and its
output is better. It runs only on the few seam frames, so it adds ~1 s to the whole job.

---

## Install

```bash
# Torch (RTX 50-series / Blackwell needs the cu128 build; adjust for your GPU/CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python-headless imageio[ffmpeg] numpy flask
# ffmpeg must be on PATH (used for concat + the slow-motion comparison)
pip install ccvfi   # (optional) v2 RIFE interpolation backend; falls back to flow if absent
```

---

## Usage

### Web app (recommended)
```bash
python webapp/server.py        # -> http://127.0.0.1:5000
```
Open the URL, **add clips in order** (drag to reorder), pick a mode, press **Run**. A live
progress bar shows each stage (analyze → render → compare) with elapsed time; when done, the
**boundary slow-motion** (raw vs fixed, side by side) and the **full result** play inline with
download links. Access it via the URL — opening the HTML file directly (`file://`) will fail on
the API calls.

### CLI
```bash
# explicit clip list, in order
python scripts/chain_normalize.py A.mp4 B.mp4 C.mp4 D.mp4 --out out/ --mode tight

# built-in sample chains
python scripts/chain_normalize.py 3            # samples/video3-{A,B,C,D}.mp4

# v2: seam motion interpolation (replaces the freeze with a real midpoint; K=1 keeps length)
python scripts/chain_normalize.py 3 --interpolate 1 --interp-backend rife
```

`--interpolate K` enables [Stage 2](#stage-2--seam-motion-interpolation-learned)
(`--interp-backend rife|flow`).

### Python API
```python
from vbf.normalize import normalize_chain

result = normalize_chain(
    ["A.mp4", "B.mp4", "C.mp4", "D.mp4"],
    out_dir="out/",
    mode="tight",            # or "balanced"
    drop_dup=True,
    interpolate=1,           # Stage 2: K midpoints per seam (0 = off)
    interp_backend="rife",   # "rife" (recommended) or "flow"
    progress=lambda stage, frac, msg: print(f"{frac:.0%} {stage} {msg}"),
)
print(result.full_path, result.slow_path, result.seams, result.seconds)
```

**Outputs** (in the chosen `out_dir`):
| file | contents |
|---|---|
| `result_full.mp4` | the whole chain, seam-normalized |
| `result_boundaries_slow.mp4` | each cut, 4× slow, **RAW ∣ FIXED** side by side |

Notes: clips are resized to the first clip's resolution; fps is taken from the first clip;
processing streams **one clip at a time** (bounded memory). A 4×~290-frame 1080×1920 chain
takes ~80 s on an RTX 5090.

---

## Repository layout
```
vbf/normalize/chain.py     # core: normalize_chain() = alignment (Stage 1) + interpolation (Stage 2)
vbf/interp/                # Stage 2 backends: flow (RAFT, no deps) + rife (ccvfi)
scripts/chain_normalize.py # CLI wrapper
webapp/                    # Flask server + static UI (slots, progress polling, players)
vbf/metrics/perceptual.py  # no-dep SSIM / Lab ΔE / motion-baseline / temporal (evaluation)
scripts/eval_boundaries.py # per-boundary metric panel (seam L1, SSIM, ΔE, downstream, ...)
scripts/interp_experiment.py # interpolation validation (self-test PSNR + ghost proxy + hold-vs-interp)
samples/                   # example generated chains (video2-*, video3-*)
```

## Design history
The seam turned out **not** to be a colour problem (the first hypothesis) nor subject motion
(the second) — a metric-vs-perception gap corrected by measuring against each clip's own motion
baseline and by full-affine analysis. The working method (chain normalization) replaced an
earlier per-seam "least-destructive cascade" that looked good on a single-frame L1 metric but
broke downstream seams and introduced a colour cast. The evaluation harness
(`scripts/eval_boundaries.py`) exists to keep that honest: it separates *did the seam get closer*
(seam L1) from *is it still perceptually intact* (SSIM, ΔE) and *did we disturb the rest*
(correction magnitude, downstream integrity).

## Limitations
- Assumes a single continuous scene per chain (shared background); hard cuts between different
  scenes are out of scope.
- Corrects **global** per-clip drift; a within-clip drift is only approximated.
- Residual subject motion is smoothed by Stage 2, but very large or heavily occluded motion
  isn't perfect (use the `rife` backend there; `flow` ghosts).
