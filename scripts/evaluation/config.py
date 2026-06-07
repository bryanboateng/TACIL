from dataclasses import dataclass
from typing import Literal

from scripts.evaluation.bryan.disturbance import DisturberConfig


@dataclass(frozen=True)
class Config:
    policy_type: Literal["dp", "bc"]
    checkpoint_artifact: str
    disturbance: DisturberConfig

    episode_seeds: tuple[int, ...] = tuple(range(1_000_000, 1_000_050))
    seed: int = 42
