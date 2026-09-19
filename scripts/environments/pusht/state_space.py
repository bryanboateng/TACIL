import torch

from scripts.environments.network_input import NetworkInput
from scripts.environments.plot_layout import OrientationGroup, PlotLayout, PositionGroup
from scripts.environments.state_space import StateSpace

AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
POSITION_SLICE = slice(0, 4)
BLOCK_ORIENTATION_INDEX = 4

STATE_DIMENSIONALITY = BLOCK_ORIENTATION_INDEX + 1
ACTION_DIMENSIONALITY = 2

ARENA_SIZE = 512

_ANGLE_INDICES = (BLOCK_ORIENTATION_INDEX,)

_AGENT_POSITION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start
_BLOCK_ORIENTATION_DIMENSIONALITY = 2
_RELATIVE_BLOCK_POSITION_DIMENSIONALITY = (
    BLOCK_POSITION_SLICE.stop - BLOCK_POSITION_SLICE.start
)
_RELATIVE_ACTION_DIMENSIONALITY = ACTION_DIMENSIONALITY

_NETWORK_INPUT_DIMENSIONALITY = (
    _AGENT_POSITION_DIMENSIONALITY
    + _BLOCK_ORIENTATION_DIMENSIONALITY
    + _RELATIVE_BLOCK_POSITION_DIMENSIONALITY
    + _RELATIVE_ACTION_DIMENSIONALITY
)

_PLOT_LAYOUT = PlotLayout(
    position_groups=(
        PositionGroup(label="Agent Position", entries=AGENT_POSITION_SLICE),
        PositionGroup(label="Block Position", entries=BLOCK_POSITION_SLICE),
    ),
    orientation_groups=(
        OrientationGroup(label="Block Orientation", index=BLOCK_ORIENTATION_INDEX),
    ),
    horizontal_limits=(0.0, float(ARENA_SIZE)),
    vertical_limits=(0.0, float(ARENA_SIZE)),
    position_unit="px",
)


def create_state_space() -> StateSpace:
    return StateSpace(
        dimensionality=STATE_DIMENSIONALITY,
        angle_indices=_ANGLE_INDICES,
        network_input=NetworkInput(
            dimensionality=_NETWORK_INPUT_DIMENSIONALITY, builder=_build_network_input
        ),
        plot_layout=_PLOT_LAYOUT,
    )


# ---- Private --------------------------------------------


def _build_network_input(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
    block_orientation_batch = state_batch[
        ..., BLOCK_ORIENTATION_INDEX : BLOCK_ORIENTATION_INDEX + 1
    ]  # (batch_size, 1)
    agent_position_batch = state_batch[..., AGENT_POSITION_SLICE]  # (batch_size, 2)
    return torch.cat(
        tensors=[
            agent_position_batch,
            torch.sin(block_orientation_batch),
            torch.cos(block_orientation_batch),
            state_batch[..., BLOCK_POSITION_SLICE] - agent_position_batch,
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )
