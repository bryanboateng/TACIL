from dataclasses import dataclass


@dataclass(frozen=True)
class WandbConfig:
    project: str
    notes: str | None = None
