from scripts.environments.plot_layout import (
    OrientationGroup,
    PlotLayout,
    PositionGroup,
    PositionUnit,
)
from scripts.environments.pusht.observation import (
    AGENT_POSITION_SLICE,
    ARENA_SIZE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
)


def create_plot_layout() -> PlotLayout:
    return PlotLayout(
        position_groups=(
            PositionGroup(label="Agent Position", entries=AGENT_POSITION_SLICE),
            PositionGroup(label="Block Position", entries=BLOCK_POSITION_SLICE),
        ),
        orientation_groups=(
            OrientationGroup(label="Block Orientation", index=BLOCK_ORIENTATION_INDEX),
        ),
        horizontal_limits=(0.0, float(ARENA_SIZE)),
        vertical_limits=(0.0, float(ARENA_SIZE)),
        position_unit=PositionUnit(label="px", scale=1.0),
    )
