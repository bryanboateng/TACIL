from scripts.environments.peg_insertion.observation import (
    GRIPPER_POSITION_SLICE,
    HOLE_ORIENTATION_INDEX,
    HOLE_POSITION_SLICE,
    PEG_ORIENTATION_INDEX,
    PEG_POSITION_SLICE,
)
from scripts.environments.plot_layout import OrientationGroup, PlotLayout, PositionGroup


def create_plot_layout() -> PlotLayout:
    return PlotLayout(
        position_groups=(
            PositionGroup(
                label="Gripper Position",
                entries=_horizontal_plane(entries=GRIPPER_POSITION_SLICE),
            ),
            PositionGroup(
                label="Peg Position",
                entries=_horizontal_plane(entries=PEG_POSITION_SLICE),
            ),
            PositionGroup(
                label="Hole Position",
                entries=_horizontal_plane(entries=HOLE_POSITION_SLICE),
            ),
        ),
        orientation_groups=(
            OrientationGroup(label="Peg Orientation", index=PEG_ORIENTATION_INDEX),
            OrientationGroup(label="Hole Orientation", index=HOLE_ORIENTATION_INDEX),
        ),
        horizontal_limits=(-0.25, 0.25),
        vertical_limits=(-0.45, 0.45),
        position_unit="m",
    )


# ---- Private --------------------------------------------


def _horizontal_plane(*, entries: slice) -> slice:
    return slice(entries.start, entries.start + 2)
