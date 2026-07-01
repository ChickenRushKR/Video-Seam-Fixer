"""CascadeRefiner — least-destructive boundary cascade on REAL B frames.

Fixes B toward A with the gentlest correction first, escalating only as needed;
generation is NOT used here (it is the documented last resort). All steps operate
on the real B frames (length preserved):

  1. color           per-channel mean match
  2. exposure/contrast  per-channel std match
  3. sharpness/grain  match B's noise level to A
  4. scale/crop       global scale+shift aligning B to A (clamped, real-frame crop)
  5. motion smoothing (opt-in; flow-based)
  6. transition blend short A_last->B cross-dissolve (4-12 frames)

Analysis compares A's last ~0.5-1s (anchor) vs B's first ~0.5-1s. Corrections are
applied to all of B (constant) by default — B's whole look matches A, no internal
drift — with an optional fade-out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vbf.data.window import BoundaryWindow
from vbf.refiners.base import BoundaryRefiner, Correction, register_refiner

_EPS = 1e-5


def _hi(x: torch.Tensor) -> torch.Tensor:
    """High-frequency residual (grain/detail) = frame - local average."""
    return x - F.avg_pool2d(x, 3, stride=1, padding=1)


def _affine_warp(frame: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta.unsqueeze(0), list(frame.shape), align_corners=True)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _backward_warp_norm(img: torch.Tensor, flow_norm: torch.Tensor) -> torch.Tensor:
    """Warp ``img`` [1,3,H,W] by a normalized flow field ``flow_norm`` [1,2,H,W] (x,y offsets)."""
    n, _, h, w = img.shape
    ys = torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    grid = base + flow_norm.permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


class CascadeCorrection(Correction):
    """Per-frame: scale/crop -> color/contrast -> grain -> (seam) blend with A_last."""

    def __init__(self, gain, bias, theta, grain_std, blend_frames, a_last, optimize_count,
                 decay=None, motion_flows=None):
        self.gain = gain                # [3]
        self.bias = bias                # [3]
        self.theta = theta              # [2,3] or None
        self.grain_std = float(grain_std)
        self.blend_frames = int(blend_frames)
        self.a_last = a_last            # [1,3,h,w] real A last frame (working res)
        self.optimize_count = optimize_count
        self.decay = decay              # [M] per-frame correction weight or None (constant)
        self.motion_flows = motion_flows  # [T,2,h,w] normalized flow B[i]->A_last, or None
        self.color_lut = None           # [3,256] histogram-match LUT (overrides gain/bias) or None
        self.color_hist_strength = 1.0  # partial-histogram blend factor
        self.local_gain_map = None      # [1,3,gh,gw] spatial lighting/color gain (overrides global color)
        self.feature_M = None           # cv2 2x3 similarity (B->A, working px) or None
        self.feature_size = None        # (h,w) the M was estimated at
        self.structure_flows = None     # [T,2,h,w] normalized smoothed flow to warp B->A locally
        self.structure_const = False    # True: apply structure_flows[0] to ALL B frames (no drift)
        self.a_tail_flows = None        # [Ka,2,h,w] warp for A's last Ka frames (bidirectional ramp)

    def apply_frame(self, frame_native: torch.Tensor, i: int) -> torch.Tensor:
        dev = frame_native.device
        w = 1.0 if self.decay is None else float(self.decay[min(i, len(self.decay) - 1)])
        f = frame_native
        if self.feature_M is not None:
            import cv2
            import numpy as np

            H, W = f.shape[-2:]
            M = self.feature_M.copy()
            M[0, 2] *= W / self.feature_size[1]
            M[1, 2] *= H / self.feature_size[0]
            img = (f[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)
            warped = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            f = torch.from_numpy(warped / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev, f.dtype)
        elif self.theta is not None and w > 0:
            th = torch.eye(2, 3, device=dev, dtype=f.dtype)
            th = th + w * (self.theta.to(dev) - torch.eye(2, 3, device=dev, dtype=f.dtype))
            f = _affine_warp(f, th)
        # local structure warp (heavily smoothed flow toward A; background only)
        if self.structure_flows is not None:
            sf = None
            if self.structure_const:
                sf = self.structure_flows[0:1]                     # same shift for all B frames (no drift)
            elif i < self.structure_flows.shape[0]:
                sf = self.structure_flows[i:i + 1]                 # bidirectional ramp
            if sf is not None:
                fl = F.interpolate(sf.to(dev), size=f.shape[-2:], mode="bilinear", align_corners=False)
                f = _backward_warp_norm(f, fl)
        if self.local_gain_map is not None:
            g = F.interpolate(self.local_gain_map.to(dev), size=f.shape[-2:], mode="bilinear", align_corners=False)
            f = (f * g).clamp(0, 1)
        elif self.color_lut is not None:
            lut = self.color_lut.to(dev)
            idx = (f.clamp(0, 1) * 255).round().long()
            mapped = torch.stack([lut[c][idx[:, c]] for c in range(3)], dim=1)
            f = f + (w * self.color_hist_strength) * (mapped - f)
        else:
            g = (1.0 + w * (self.gain.to(dev) - 1.0)).view(1, 3, 1, 1)
            b = (w * self.bias.to(dev)).view(1, 3, 1, 1)
            f = f * g + b
        if self.grain_std > 0 and w > 0:
            f = f + torch.randn_like(f) * (self.grain_std * w)
        # step 5: motion-compensated transition — A flow-aligned to B's layout, then
        # dissolve appearance only (structure matches -> no afterimage)
        if self.motion_flows is not None and i < self.motion_flows.shape[0]:
            a = F.interpolate(self.a_last.to(dev), size=f.shape[-2:], mode="bilinear", align_corners=False)
            flow = F.interpolate(self.motion_flows[i:i + 1].to(dev), size=f.shape[-2:], mode="bilinear", align_corners=False)
            a_aligned = _backward_warp_norm(a, flow)
            alpha = (i + 1) / (self.motion_flows.shape[0] + 1)
            f = (1 - alpha) * a_aligned + alpha * f
        elif i < self.blend_frames:
            a = F.interpolate(self.a_last.to(dev), size=f.shape[-2:], mode="bilinear", align_corners=False)
            alpha = (i + 1) / (self.blend_frames + 1)
            f = (1 - alpha) * a + alpha * f
        return f.clamp(0, 1)

    def set_fit(self):
        """Snapshot the fitted color/geometry so scale_strength can interpolate to identity."""
        import numpy as np
        self._fit = (
            self.gain.clone(), self.bias.clone(),
            None if self.theta is None else self.theta.clone(),
            None if self.feature_M is None else self.feature_M.copy(),
            float(self.grain_std),
        )
        _ = np  # keep import local

    def scale_strength(self, s: float) -> None:
        """Scale the whole correction toward identity by ``s`` in [0,1] (0 = passthrough).

        Interpolates every fitted parameter from identity (s=0) to its fitted value
        (s=1): gain->1, bias->0, affine/feature->I, grain->0. Requires ``set_fit`` first.
        """
        import numpy as np

        g0, b0, th0, fm0, gr0 = self._fit
        self.gain = 1.0 + s * (g0 - 1.0)
        self.bias = s * b0
        if th0 is not None:
            I = torch.eye(2, 3, device=th0.device, dtype=th0.dtype)
            self.theta = I + s * (th0 - I)
        if fm0 is not None:
            I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=fm0.dtype)
            self.feature_M = I + s * (fm0 - I)
        self.grain_std = s * gr0

    def apply_a_frame(self, frame_native: torch.Tensor, j: int) -> torch.Tensor:
        """Warp A's tail frame toward B (bidirectional ramp); geometric only, A keeps its color."""
        if self.a_tail_flows is None or j >= self.a_tail_flows.shape[0]:
            return frame_native
        dev = frame_native.device
        fl = F.interpolate(self.a_tail_flows[j:j + 1].to(dev), size=frame_native.shape[-2:],
                           mode="bilinear", align_corners=False)
        return _backward_warp_norm(frame_native, fl).clamp(0, 1)


@register_refiner("cascade")
class CascadeRefiner(BoundaryRefiner):
    @staticmethod
    def _scene_motion(window) -> float:
        """Normal adjacent-frame change of the scene (A tail + B head, seam excluded)."""
        a, b = window.anchor, window.b_original[:18]
        diffs = []
        if a.shape[0] > 1:
            diffs.append((a[1:] - a[:-1]).abs().flatten(1).mean(1))
        if b.shape[0] > 1:
            diffs.append((b[1:] - b[:-1]).abs().flatten(1).mean(1))
        if not diffs:
            return 0.0
        return float(torch.cat(diffs).mean())

    def _strength_to_baseline(self, window, corr, threshold: float) -> float:
        """Smallest strength s in [0,1] whose corrected seam L1 <= motion_baseline*threshold.

        Returns 0.0 (passthrough) when the raw seam is already at/below baseline — the
        honest answer when B is already continuous with A (no over-correction below the
        scene's own motion floor)."""
        a1, b1 = window.anchor[-1:], window.b_original[:1]
        target = self._scene_motion(window) * threshold
        raw = float((b1 - a1).abs().mean())
        if raw <= target:
            return 0.0
        for s in torch.linspace(0.0, 1.0, 11).tolist():
            corr.scale_strength(s)
            if float((corr.apply_frame(b1.clone(), 0) - a1).abs().mean()) <= target:
                return s
        return 1.0

    def _flow_estimator(self):
        est = self.loss_fn.metrics.get("flow")
        if est is None:
            from vbf.metrics.flow import FlowMetric

            est = FlowMetric(self.config.loss.metrics.flow, self.device)
        return est

    _seg = None

    @torch.no_grad()
    def _subject_bg_mask(self, window, blur_k):
        """Background weight map [1,1,h,w] (~1 background, ~0 subject). Uses person
        segmentation (robust even when the subject is static); falls back to within-A
        motion if seg is unavailable. Protects the subject from warp/recolour."""
        ccfg = self.config.refiner.cascade
        dev, dt = window.anchor.device, window.anchor.dtype
        h, w = window.anchor.shape[-2:]
        frames = torch.cat([window.anchor[-1:], window.b_original[:1]], 0)  # A_last, B0
        try:
            if CascadeRefiner._seg is None:
                from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
                CascadeRefiner._seg = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval().to(dev)
            mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=dt).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=dt).view(1, 3, 1, 1)
            out = CascadeRefiner._seg(((frames - mean) / std))["out"].softmax(1)[:, 15]  # VOC person=15
            person = out.amax(0, keepdim=True).unsqueeze(0)        # [1,1,h,w] union over A_last,B0
            person = F.max_pool2d(person, 25, stride=1, padding=12)  # dilate to protect edges
            person = F.avg_pool2d(person, blur_k | 1, stride=1, padding=(blur_k | 1) // 2)  # feather
            return (1.0 - person.clamp(0, 1))
        except Exception:
            if window.anchor.shape[0] >= 2:
                vm = self._flow_estimator()._estimate_flow(window.anchor[-2:-1], window.anchor[-1:])[0]
                vm = vm.pow(2).sum(0).clamp_min(1e-12).sqrt()
                vm = F.avg_pool2d(vm[None, None], blur_k | 1, stride=1, padding=(blur_k | 1) // 2)[0, 0]
                return (1.0 / (1.0 + (vm / (2 * (vm.median() + 1e-3))) ** 2)).view(1, 1, h, w)
            return torch.ones(1, 1, h, w, device=dev, dtype=dt)

    @staticmethod
    def _robust_line(p, d, iters=5):
        """Robust fit d = slope*p + intercept (IRLS). Returns (slope, intercept)."""
        wts = torch.ones_like(p)
        slope = intercept = 0.0
        for _ in range(iters):
            X = torch.stack([p, torch.ones_like(p)], dim=1)
            XtW = (X * wts.unsqueeze(1)).T
            sol = torch.linalg.solve(XtW @ X + 1e-6 * torch.eye(2, device=p.device, dtype=p.dtype), XtW @ d)
            slope, intercept = float(sol[0]), float(sol[1])
            resid = (X @ sol - d).abs()
            sigma = resid.median() + 1e-6
            wts = 1.0 / (1.0 + (resid / (2 * sigma)) ** 2)
        return slope, intercept

    @torch.no_grad()
    def _fit_scale_crop_feature(self, a_last, b0, ccfg):
        """ORB+RANSAC similarity (scale/rot/shift) aligning B to A's static background.
        RANSAC rejects the moving subject as outliers -> precise framing match."""
        import cv2
        import numpy as np

        def gray(t):
            x = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

        A, B = gray(a_last), gray(b0)
        h, w = A.shape
        orb = cv2.ORB_create(3000)
        ka, da = orb.detectAndCompute(A, None)
        kb, db = orb.detectAndCompute(B, None)
        if da is None or db is None or len(ka) < 12 or len(kb) < 12:
            return None, None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)  # query B -> train A
        if len(matches) < 12:
            return None, None
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        src = np.float32([kb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # B
        dst = np.float32([ka[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # A
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return None, None
        s = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
        if abs(s - 1) > 2 * ccfg.max_scale:               # reject wild scale
            return None, None
        # empirical check: does aligning B by M get closer to A?
        an = a_last[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        bn = b0[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        bc = cv2.warpAffine(bn.astype(np.float32), M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if np.abs(an - bc).mean() >= np.abs(an - bn).mean():
            return None, None
        return M, (h, w)

    @torch.no_grad()
    def _fit_scale_crop(self, a_last, b0, ccfg):
        """Robust ANISOTROPIC scale (sx,sy) + shift aligning B to A; empirically sign-checked."""
        est = self._flow_estimator()
        h, w = a_last.shape[-2:]
        dev, dt = a_last.device, a_last.dtype
        flow = est._estimate_flow(a_last, b0)[0]            # [2,H,W] px (A->B)
        disp = torch.stack([flow[0] * (2.0 / max(w - 1, 1)), flow[1] * (2.0 / max(h - 1, 1))])
        disp = F.adaptive_avg_pool2d(disp.unsqueeze(0), (24, 24))[0]
        ys = torch.linspace(-1, 1, 24, device=dev, dtype=dt)
        xs = torch.linspace(-1, 1, 24, device=dev, dtype=dt)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        px, py = gx.reshape(-1), gy.reshape(-1)
        ax, tx = self._robust_line(px, disp[0].reshape(-1))   # dx = ax*px + tx
        ay, ty = self._robust_line(py, disp[1].reshape(-1))   # dy = ay*py + ty
        clamp = lambda v, m: max(-m, min(m, v))
        sx = 1 + clamp(ax, ccfg.max_scale)
        sy = 1 + clamp(ay, ccfg.max_scale)
        tx, ty = clamp(tx, ccfg.max_shift), clamp(ty, ccfg.max_shift)
        theta = torch.tensor([[sx, 0, tx], [0, sy, ty]], device=dev, dtype=dt)
        theta_inv = torch.tensor([[1 / sx, 0, -tx / sx], [0, 1 / sy, -ty / sy]], device=dev, dtype=dt)
        base = (b0 - a_last).abs().mean()
        best = min((base, None), ((_affine_warp(b0, theta) - a_last).abs().mean(), theta),
                   ((_affine_warp(b0, theta_inv) - a_last).abs().mean(), theta_inv), key=lambda t: float(t[0]))
        return best[1]

    @torch.no_grad()
    def refine(self, window: BoundaryWindow) -> tuple[BoundaryWindow, Correction]:
        ccfg = self.config.refiner.cascade
        window.to(self.device)
        dev, dt = window.anchor.device, window.anchor.dtype
        m = window.optimize_count

        a_tail = window.anchor                                   # real A frames (working res)
        nb = min(ccfg.analysis_frames, window.b_original.shape[0])
        b_head = window.b_original[:nb]                          # real B head

        af = a_tail.flatten(2)                                   # [Ka,3,*]
        bf = b_head.flatten(2)
        a_mean, a_std = af.mean((0, 2)), af.std((0, 2))          # [3]
        b_mean, b_std = bf.mean((0, 2)), bf.std((0, 2))

        gain = (a_std / (b_std + _EPS)).clamp(0.5, 2.0) if ccfg.contrast else torch.ones(3, device=dev, dtype=dt)
        target_mean = a_mean if ccfg.color else b_mean
        bias = target_mean - gain * b_mean

        grain_std = 0.0
        if ccfg.grain:
            a_g = float(_hi(a_tail).std())
            b_g = float(_hi(b_head).std())
            if a_g > b_g:                                        # B smoother than A -> add grain
                grain_std = (a_g ** 2 - b_g ** 2) ** 0.5

        theta = None
        feature_M = feature_size = None
        if ccfg.scale_crop:
            if ccfg.scale_method == "feature":
                feature_M, feature_size = self._fit_scale_crop_feature(window.anchor[-1:], window.b_original[:1], ccfg)
            else:
                theta = self._fit_scale_crop(window.anchor[-1:], window.b_original[:1], ccfg)

        decay = None
        if ccfg.apply_decay:
            idx = torch.arange(m, device=dev, dtype=dt)
            decay = (1.0 - idx / max(m - 1, 1)).clamp_min(0.0)

        structure_flows = None
        a_tail_flows = None
        structure_const = False
        bg_mask4 = None
        if ccfg.structure_warp:
            est = self._flow_estimator()
            a_last = window.anchor[-1:]
            h, w = a_last.shape[-2:]
            k = ccfg.structure_blur | 1  # odd
            pad = k // 2

            def _blur(fl):
                return F.avg_pool2d(fl.unsqueeze(0), k, stride=1, padding=pad)[0]

            def _norm(fl):
                n = torch.stack([fl[0] * (2.0 / max(w - 1, 1)), fl[1] * (2.0 / max(h - 1, 1))])
                return n.clamp(-ccfg.structure_max_disp, ccfg.structure_max_disp)

            b0 = window.b_original[:1]
            fn0 = _norm(_blur(est._estimate_flow(b0, a_last)[0]))  # B0->A_last unit (unsigned)
            base = (b0 - a_last).abs().mean()
            ep = (_backward_warp_norm(b0, fn0.unsqueeze(0)) - a_last).abs().mean()
            em = (_backward_warp_norm(b0, -fn0.unsqueeze(0)) - a_last).abs().mean()
            sign = min((base, 0.0), (ep, 1.0), (em, -1.0), key=lambda t: float(t[0]))[1]

            # subject mask: warp/recolour ONLY the background; protect the subject
            # (person segmentation -> robust even when the subject is static).
            if ccfg.subject_mask:
                bg_mask4 = self._subject_bg_mask(window, k)   # [1,1,h,w]
                bg_mask = bg_mask4[0]                          # [1,h,w]
            else:
                bg_mask4 = None
                bg_mask = torch.ones(1, h, w, device=dev, dtype=dt)

            if sign != 0.0:
                F0 = fn0 * sign * bg_mask                  # full unit flow B->A, background only
                if ccfg.bidirectional:
                    # split the warp: B head -> halfway toward A; A tail -> halfway toward B;
                    # each ramps to 0 away from the seam (frame-by-frame ratio change).
                    Kb = min(ccfg.ramp_b_frames, window.b_original.shape[0])
                    Ka = min(ccfg.ramp_a_frames, window.anchor.shape[0])
                    bflows = []
                    for i in range(Kb):
                        Fi = _norm(_blur(est._estimate_flow(window.b_original[i:i + 1], a_last)[0])) * sign * bg_mask
                        bflows.append(Fi * 0.5 * (1.0 - i / max(Kb - 1, 1)))
                    structure_flows = torch.stack(bflows).cpu()
                    aflows = [(-F0) * 0.5 * (idx / max(Ka - 1, 1)) for idx in range(Ka)]  # A[-Ka+idx]
                    a_tail_flows = torch.stack(aflows).cpu()
                else:
                    # CONSTANT: shift B's background to A's position for the WHOLE clip
                    # (no decay/ramp -> static background doesn't drift; seam aligns).
                    structure_flows = (F0 * ccfg.structure_strength).unsqueeze(0).cpu()
                    structure_const = True

        local_gain_map = None
        if ccfg.local_color:
            est = self._flow_estimator()
            a_last = window.anchor[-1:]
            h, w = a_last.shape[-2:]
            b0a = window.b_original[:1]
            if structure_flows is not None and structure_const:  # align B's background to A first
                b0a = _backward_warp_norm(b0a, structure_flows[0:1].to(dev))
            kk = ccfg.local_color_blur | 1
            pp = kk // 2
            a_lf = F.avg_pool2d(a_last, kk, stride=1, padding=pp)
            b_lf = F.avg_pool2d(b0a, kk, stride=1, padding=pp)
            lg = (a_lf / (b_lf + 1e-3)).clamp(1.0 / ccfg.local_color_clamp, ccfg.local_color_clamp)  # [1,3,h,w]
            if ccfg.subject_mask:  # don't relight the subject (person-seg mask)
                bgm = bg_mask4 if bg_mask4 is not None else self._subject_bg_mask(window, kk)
            else:
                bgm = 1.0
            lg = 1.0 + bgm * ccfg.local_color_strength * (lg - 1.0)
            local_gain_map = F.adaptive_avg_pool2d(lg, (max(8, h // 8), max(8, w // 8))).cpu()  # smooth, small

        motion_flows = None
        if ccfg.motion_smooth:
            est = self._flow_estimator()
            a_last = window.anchor[-1:]
            h, w = a_last.shape[-2:]
            flows = []
            for i in range(min(ccfg.motion_frames, window.b_original.shape[0])):
                fpx = est._estimate_flow(window.b_original[i:i + 1], a_last)[0]  # B[i]->A_last, px
                flows.append(torch.stack([fpx[0] * (2.0 / max(w - 1, 1)), fpx[1] * (2.0 / max(h - 1, 1))]))
            motion_flows = torch.stack(flows).cpu()  # [T,2,h,w] normalized

        # build corrected working B for logging/snapshot
        corr = CascadeCorrection(gain, bias, theta, grain_std, ccfg.blend_frames,
                                 window.anchor[-1:].cpu(), m, decay, motion_flows)
        corr.feature_M, corr.feature_size = feature_M, feature_size
        corr.structure_flows = structure_flows
        corr.structure_const = structure_const
        corr.local_gain_map = local_gain_map
        corr.a_tail_flows = a_tail_flows
        corr.a_tail_count = 0 if a_tail_flows is None else a_tail_flows.shape[0]

        a1, b1 = window.anchor[-1:], window.b_original[:1]
        ones3, zeros3 = torch.ones(3, device=dev, dtype=dt), torch.zeros(3, device=dev, dtype=dt)

        if ccfg.conservative:
            # CONSERVATIVE: components already limited to the least-destructive set (mean colour,
            # weak scale/crop, grain), faded to zero across the window (apply_decay + optimize_b),
            # so B's tail (=C's generation anchor) is untouched. Choose HOW MUCH to apply, and
            # only accept it if it does not perceptually hurt the seam vs raw.
            from vbf.metrics.perceptual import delta_e76, ssim

            corr.set_fit()
            base = self._scene_motion(window)
            raw_l1 = float((b1 - a1).abs().mean())
            if ccfg.strength_search:
                s = self._strength_to_baseline(window, corr, ccfg.gate_threshold)
                corr.scale_strength(s)
            else:
                s = 1.0
            # acceptance gate: reject if structure (SSIM) or colour (ΔE) got worse than raw
            raw_ssim, raw_de = ssim(a1[0], b1[0]), delta_e76(a1[0], b1[0])
            cb = corr.apply_frame(b1.clone(), 0)
            cor_ssim, cor_de = ssim(a1[0], cb[0]), delta_e76(a1[0], cb[0])
            accepted = not (cor_ssim < raw_ssim - 1e-3 or cor_de > raw_de + 1e-2)
            if not accepted:
                corr.scale_strength(0.0)
                s = 0.0
            if self.logger is not None:
                self.logger.log_text(
                    "conservative",
                    f"strength={s:.2f} accepted={accepted} baseline={base:.4f} raw_l1={raw_l1:.4f} "
                    f"(ratio={raw_l1 / base if base else float('nan'):.2f}) "
                    f"ssim {raw_ssim:.3f}->{cor_ssim:.3f} dE {raw_de:.2f}->{cor_de:.2f}",
                )
        else:
            # AUTO-SELECT (scene-adaptive): keep only the components that actually reduce the
            # seam gap. structure/local-colour help a stable scene (tunnel) but can HURT a
            # reflective/dynamic one (riverside) where flow is unreliable -> auto-drop them.
            full = dict(feature_M=corr.feature_M, structure_flows=corr.structure_flows,
                        a_tail_flows=corr.a_tail_flows, local_gain_map=corr.local_gain_map,
                        color_lut=corr.color_lut, gain=corr.gain, bias=corr.bias)

            def apply_cfg(cfg):
                corr.feature_M = cfg.get("feature_M", full["feature_M"])
                corr.structure_flows = cfg.get("structure_flows", full["structure_flows"])
                corr.a_tail_flows = cfg.get("a_tail_flows", full["a_tail_flows"])
                corr.a_tail_count = 0 if corr.a_tail_flows is None else corr.a_tail_flows.shape[0]
                corr.local_gain_map = cfg.get("local_gain_map", full["local_gain_map"])
                corr.color_lut = cfg.get("color_lut", full["color_lut"])
                corr.gain = cfg.get("gain", full["gain"])
                corr.bias = cfg.get("bias", full["bias"])

            NONE_GEO = dict(structure_flows=None, a_tail_flows=None, local_gain_map=None)
            NONE_COL = dict(color_lut=None, gain=ones3, bias=zeros3)
            candidates = [
                ("raw", {**NONE_GEO, **NONE_COL, "feature_M": None}),
                ("feat", {**NONE_GEO, **NONE_COL}),
                ("feat+color", {**NONE_GEO}),
                ("feat+struct+color", dict(local_gain_map=None)),
                ("full", {}),
            ]
            scored = []
            for name, cfg in candidates:
                apply_cfg(cfg)
                scored.append((float((corr.apply_frame(b1.clone(), 0) - a1).abs().mean()), name, cfg))
            best_gap, best_name, best_cfg = min(scored, key=lambda t: t[0])
            apply_cfg(best_cfg)
            if self.logger is not None:
                self.logger.log_text("auto_select", f"chosen={best_name} gap={best_gap:.4f} | " +
                                     " ".join(f"{n}:{g:.4f}" for g, n, _ in scored))

        if ccfg.color_histogram:  # finer color: per-channel histogram-match LUT (B distribution -> A)
            qs = torch.linspace(0, 1, 256, device=dev, dtype=dt)
            def _sub(x):
                return x if x.numel() <= 200000 else x[torch.randint(0, x.numel(), (200000,), device=x.device)]
            lut = torch.empty(3, 256, device=dev, dtype=dt)
            for c in range(3):
                bq = torch.quantile(_sub(bf[:, c].reshape(-1)), qs)
                aq = torch.quantile(_sub(af[:, c].reshape(-1)), qs)
                idx = torch.searchsorted(bq.contiguous(), qs).clamp(0, 255)
                lut[c] = aq[idx]
            corr.color_lut = lut.cpu()
            corr.color_hist_strength = ccfg.color_hist_strength
        if self.logger is not None:
            with torch.no_grad():
                b_corr = torch.stack([corr.apply_frame(window.b_original[i:i + 1], i)[0] for i in range(min(m, 24))])
                seq = torch.cat([window.anchor, b_corr], dim=0)
                metrics = self.loss_fn.observe(window, seq)
                metrics.update({"cascade/gain": float(gain.mean()), "cascade/bias": float(bias.mean()),
                                "cascade/grain": grain_std,
                                "cascade/scale": float(theta[0, 0]) if theta is not None else 1.0})
                window.b = torch.cat([b_corr, window.b[min(m, 24):]], 0) if window.b.shape[0] > 24 else b_corr
            self.logger.log_step(step=0, losses={"loss/total": 0.0}, metrics=metrics, window=window)

        return window, corr
