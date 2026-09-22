from scripts.environments.peg_insertion.observation import (
    HOLE_ORIENTATION_INDEX,
    HOLE_POSITION_SLICE,
    PEG_POSITION_SLICE,
    TCP_POSITION_SLICE,
)
from scripts.environments.plot_layout import (
    OrientationGroup,
    PlotLayout,
    PositionGroup,
    PositionUnit,
)


def create_plot_layout() -> PlotLayout:
    return PlotLayout(
        position_groups=(
            PositionGroup(
                label="TCP Position",
                entries=_horizontal_plane(entries=TCP_POSITION_SLICE),
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
            OrientationGroup(label="Hole Orientation", index=HOLE_ORIENTATION_INDEX),
        ),
        horizontal_limits=(-200.0, 180.0),
        vertical_limits=(-370.0, 420.0),
        position_unit=PositionUnit(label="mm", scale=1000.0),
    )


# ---- Private --------------------------------------------


def _horizontal_plane(*, entries: slice) -> slice:
    return slice(entries.start, entries.start + 2)
