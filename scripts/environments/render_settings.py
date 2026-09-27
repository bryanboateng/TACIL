from dataclasses import dataclass


@dataclass(frozen=True)
class RenderSettings:
    size: int
    supersample_factor: int = 1
