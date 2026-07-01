"""Metric interface + registry.

Every observable quantity (flicker, ratio, flow continuity) implements the same
interface so it can be both *logged* (``observe`` -> scalars) and used as a
*differentiable loss* (``loss`` -> Tensor). The registry lets config select and
weight metrics by name, and lets future refiners (e.g. RL) reuse the exact same
metrics as rewards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from vbf.data.window import BoundaryWindow

REGISTRY: dict[str, type["Metric"]] = {}


def register(name: str) -> Callable[[type["Metric"]], type["Metric"]]:
    def deco(cls: type["Metric"]) -> type["Metric"]:
        if name in REGISTRY:
            raise ValueError(f"metric '{name}' already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def build_metric(name: str, cfg: object, device: torch.device | str) -> "Metric":
    if name not in REGISTRY:
        raise KeyError(f"unknown metric '{name}'. registered: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, device)


class Metric(ABC):
    """Base metric. Operates on the assembled sequence of a BoundaryWindow.

    Implementations receive the assembled sequence ``seq`` ([T,3,H,W], may carry
    gradient w.r.t. optimizable frames) and the ``boundary_index`` (index of the
    first B frame; the seam is between ``boundary_index-1`` and ``boundary_index``).
    """

    name: str = "metric"

    def __init__(self, cfg: object, device: torch.device | str):
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def loss(self, seq: torch.Tensor, boundary_index: int) -> torch.Tensor:
        """Differentiable scalar loss (lower = smoother boundary)."""

    @abstractmethod
    def observe(self, seq: torch.Tensor, boundary_index: int) -> dict[str, float]:
        """Scalar diagnostics for logging (no grad needed)."""

    def loss_for(self, window: BoundaryWindow, seq: torch.Tensor) -> torch.Tensor:
        return self.loss(seq, window.boundary_index)

    def observe_for(self, window: BoundaryWindow, seq: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            return self.observe(seq, window.boundary_index)
