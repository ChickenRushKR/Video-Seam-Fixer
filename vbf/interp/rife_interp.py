"""RIFE frame interpolation backend (opt-in).

Learned interpolator (RIFE v4.26-heavy via the ``ccvfi`` package, weights auto-downloaded
from HuggingFace). Unlike the flow-warp backend it handles occlusion / fast motion (e.g. a
quickly-moving hand at a seam) without the torn "ghost". Same interface as
``FlowInterpolator``: ``interpolate(f0, f1, t) -> f_t``.

Requires ``pip install ccvfi`` (+ network on first run to fetch weights). ``get_interpolator``
falls back to the flow backend if this import fails.
"""

from __future__ import annotations

import torch


class RifeInterpolator:
    backend = "rife"

    def __init__(self, device=None, scale: float = 1.0, fp16: bool = True):
        from ccvfi import AutoModel, ConfigType

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scale = scale
        self.fp16 = fp16 and self.device.type == "cuda"
        self.model = AutoModel.from_pretrained(
            ConfigType.RIFE_IFNet_v426_heavy, device=self.device, fp16=self.fp16
        )
        self._dtype = torch.float16 if self.fp16 else torch.float32

    @torch.no_grad()
    def interpolate(self, f0: torch.Tensor, f1: torch.Tensor, t: float) -> torch.Tensor:
        imgs = torch.stack([f0, f1], dim=0).unsqueeze(0)          # (1,2,C,H,W)
        imgs = imgs.to(self.device, self._dtype)
        out = self.model.inference(imgs, float(t), self.scale)   # (1,C,H,W)
        return out[0].float().clamp(0, 1).cpu()

    @torch.no_grad()
    def occlusion_score(self, f0: torch.Tensor, f1: torch.Tensor) -> float:
        # RIFE resolves occlusion internally; expose 0.0 (parity with the flow interface).
        return 0.0
