from dataclasses import dataclass


@dataclass(frozen=True)
class WindowedLayout:
    """The consumer trains on windows of this length, not on single pairs.

    An augmentation that gets this must give every synthetic state enough
    expert continuation behind it to fill one window.
    """

    prediction_horizon: int
