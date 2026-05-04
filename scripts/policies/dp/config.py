"""Shared configuration for the PushT diffusion policy.

Both training (``train.py``) and inference (``policy.py``) read from here so
the two stay in lockstep. Do not duplicate these values elsewhere.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    seed: int

    observation_dim: int = 5
    action_dim: int = 2

    prediction_horizon: int = 16
    observation_horizon: int = 2
    action_horizon: int = 8

    num_diffusion_steps: int = 100

    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    num_warmup_steps: int = 500
    ema_power: float = 0.75
    num_dataloader_workers: int = 1

    @property
    def global_cond_dim(self) -> int:
        return self.observation_dim * self.observation_horizon
