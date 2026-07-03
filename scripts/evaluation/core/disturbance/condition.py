import colorsys
from dataclasses import dataclass

from matplotlib.colors import Colormap, LinearSegmentedColormap


@dataclass(frozen=True)
class Condition:
    # Floor damping passed to the PushT env. None uses the env default (quasi-static,
    # high-friction floor); larger values let the block slide further.
    damping: float | None
    # Block-push disturbance applied on top of the dynamics. None means no push.
    push: float | None


DAMPING_LEVELS: list[float | None] = [None]  # , 0.4, 0.7]
PUSH_MAGNITUDES: list[float | None] = [200.0, 500.0, 800.0]


def label(condition: Condition) -> str:
    if condition.damping is None and condition.push is None:
        return "undisturbed"

    damping_part = (
        "damp_default" if condition.damping is None else f"damp{condition.damping}"
    )
    push_part = "no_push" if condition.push is None else f"push{int(condition.push)}"
    return f"{damping_part}__{push_part}"


# Each condition renders in a single-hue palette so both disturbance axes can be read
# straight off a video. The *hue* encodes the push magnitude (cool -> hot as the shove
# grows); the *depth* (darkness + saturation) encodes the damping level (the slidier the
# floor, the deeper the color). Keep the keys aligned with PUSH_MAGNITUDES/DAMPING_LEVELS.
_PUSH_HUE_AND_SATURATION: dict[float | None, tuple[float, float]] = {
    None: (210.0, 0.16),  # calm blue-grey: no push
    200.0: (150.0, 0.45),  # teal-green
    500.0: (32.0, 0.75),  # orange
    800.0: (5.0, 0.95),  # red
}
_DAMPING_BRIGHTNESS: dict[float | None, float] = {
    None: 0.86,  # light: default quasi-static floor
    0.4: 0.62,
    0.7: 0.42,  # dark: slidiest floor
}


def condition_colormap(condition: Condition) -> Colormap:
    magnitude = None if condition.push is None else condition.push
    hue, saturation = _PUSH_HUE_AND_SATURATION[magnitude]
    brightness = _DAMPING_BRIGHTNESS[condition.damping]

    base = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
    # A light, desaturated tint of the same hue anchors the low end so the renderer's
    # light-to-dark sample stops stay a coherent single-hue ramp.
    light = colorsys.hsv_to_rgb(
        hue / 360.0, saturation * 0.25, min(1.0, brightness + 0.30)
    )
    return LinearSegmentedColormap.from_list(label(condition), colors=[light, base])


def build_conditions() -> list[Condition]:
    return [
        Condition(damping=damping, push=magnitude)
        for damping in DAMPING_LEVELS
        for magnitude in PUSH_MAGNITUDES
    ]
