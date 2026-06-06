from dataclasses import dataclass
from pathlib import Path


@dataclass
class Stat:
    max_coverage_mean: float
    epoch: int
    state_dict_path: Path
