# -*- coding: utf-8 -*-
"""Paper 5+8 ablation runtime — Colab-friendly drop-in for world_model imports.

Provides:
    - SCENARIO_DIM, SCENARIOS (with .id attribute)
    - LatentDynamicsModule (real nn.Module, supports .to(device))
    - _build_model(z_dim, scenario_dim, n_context) returning the module

This avoids importing the GIS Data Agent project (D:/adk/data_agent/world_model.py)
which is not on the Colab filesystem and pulls in heavy unused deps.

The architecture is byte-for-byte equivalent to data_agent/world_model.py:184-287
so checkpoints saved by either side load into the other (modulo z_dim).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

Z_DIM = 64
SCENARIO_DIM = 16
N_CONTEXT = 2


@dataclass
class _Scenario:
    id: int
    name: str


SCENARIOS = {
    "urban_sprawl":                _Scenario(0, "urban_sprawl"),
    "ecological_restoration":      _Scenario(1, "ecological_restoration"),
    "agricultural_intensification":_Scenario(2, "agricultural_intensification"),
    "climate_adaptation":          _Scenario(3, "climate_adaptation"),
    "baseline":                    _Scenario(4, "baseline"),
}


class LatentDynamicsModule(nn.Module):
    """Residual CNN: z_{t+1} = z_t + f(z_t, s, ctx).

    Mirrors data_agent/world_model.py:_LatentDynamicsModule exactly.
    """

    def __init__(self, z_dim: int = Z_DIM, scenario_dim: int = SCENARIO_DIM, n_context: int = N_CONTEXT):
        super().__init__()
        self.z_dim = z_dim
        self.scenario_dim = scenario_dim
        self.n_context = n_context
        self.scenario_enc = nn.Sequential(
            nn.Linear(scenario_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim),
        )
        in_channels = z_dim * 2 + n_context
        self.dynamics = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, dilation=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=4, dilation=4),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, z_dim, 1),
        )

    def forward(self, z_t, scenario, context=None):
        s = self.scenario_enc(scenario)[:, :, None, None].expand_as(z_t)
        if context is not None:
            inp = torch.cat([z_t, s, context], dim=1)
        else:
            B, _, H, W = z_t.shape
            zeros = torch.zeros(B, self.n_context, H, W, device=z_t.device)
            inp = torch.cat([z_t, s, zeros], dim=1)
        delta_z = self.dynamics(inp)
        return z_t + delta_z


def _build_model(z_dim: int = Z_DIM, scenario_dim: int = SCENARIO_DIM, n_context: int = N_CONTEXT) -> LatentDynamicsModule:
    return LatentDynamicsModule(z_dim, scenario_dim, n_context)
