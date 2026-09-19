from scripts.environments.state_space import StateSpace

AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
POSITION_SLICE = slice(0, 4)
BLOCK_ORIENTATION_INDEX = 4

STATE_DIMENSIONALITY = BLOCK_ORIENTATION_INDEX + 1
ACTION_DIMENSIONALITY = 2

ARENA_SIZE = 512

_ANGLE_INDICES = (BLOCK_ORIENTATION_INDEX,)


def create_state_space() -> StateSpace:
    return StateSpace(dimensionality=STATE_DIMENSIONALITY, angle_indices=_ANGLE_INDICES)
