"""Checks for configuration values, phrased so the field names them.

Every check takes its values as keyword arguments, so the message says which
field was wrong without the caller having to repeat its name:

    check_positive(window_length=self.window_length)
    -> ValueError: window_length must be positive, but it is 0.

A configuration is read from the command line, and a sweep draws it without
anyone looking. A value that makes no sense should stop the run where it is
set, rather than surface much later as a division by zero, an empty network,
or a silently useless result.
"""

from collections.abc import Sequence


def check_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, but it is {value}.")


def check_not_negative(**values: float) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must not be negative, but it is {value}.")


def check_share(**values: float) -> None:
    """A share of something, so between zero and one inclusive."""
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, but it is {value}.")


def check_positive_sizes(**sequences: Sequence[int]) -> None:
    """A sequence of layer sizes: it needs at least one, and none may vanish."""
    for name, sizes in sequences.items():
        if len(sizes) == 0:
            raise ValueError(f"{name} must name at least one size, but it is empty.")
        for index, size in enumerate(iterable=sizes):
            if size <= 0:
                raise ValueError(
                    f"{name} must hold positive sizes, but entry {index} is {size}."
                )


def check_not_empty(**values: str) -> None:
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"{name} must not be empty.")
