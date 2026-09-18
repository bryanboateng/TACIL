from typing import Protocol


class Condition(Protocol):
    @property
    def label(self) -> str: ...
