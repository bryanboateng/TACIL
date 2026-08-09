"""What the numbers in a PushT state mean.

A state is four positions and one orientation. The orientation is periodic, so
the state is not a plain vector: 0.01 and 2π - 0.01 are almost the same
orientation but far apart as numbers. Only this one fact makes the module
necessary. Everything below follows from it.

The remedy is to leave the state alone and to replace the three arithmetic
operations that would otherwise be wrong at that seam:

    a - b        ->  state_differences(a, b)
    a + d        ->  advance_states(a, d)
    x.mean(dim)  ->  average_states(x, dim)

Every stage then keeps working on flat vectors. Hertzberg et al. call the first
two operators ⊟ and ⊞ and name exactly this pattern; see "Integrating generic
sensor fusion algorithms with sound state representations through encapsulation
of manifolds", Information Fusion 14(1), 2013, arXiv:1107.1119. average_states
is the matching circular mean from directional statistics.

The state carries no velocity, and it does not need one. With gym_pusht's
default damping the block's velocity is destroyed every substep, so the next
state is a function of this state and the action alone. Measured: a block given
150 px/s stands still one step later.

This module holds the layout and nothing built on top of it. How a network is
fed from such a state is TaCIL's business and lives in
scripts/data/augmentation/tacil/models/network_input.py. Both TaCIL and the GPI
baseline read the layout from here, so the import graph shows who depends on
PushT.

The widths are deliberately absent. Every stage reads them from the recorded
data, which states them itself, so no policy has to name the environment.
"""

import torch

# Layout: agent x/y, block x/y, block angle.
AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
POSITION_SLICE = slice(0, 4)
ANGLE_INDEX = 4

# The parts a state error is broken down into.
STATE_GROUPS = (
    ("agent", AGENT_POSITION_SLICE),
    ("block", BLOCK_POSITION_SLICE),
    ("angle", slice(ANGLE_INDEX, ANGLE_INDEX + 1)),
)

# The quadratic arena (gym_pusht window size, pixels).
ARENA_SIZE = 512

FULL_TURN = 2.0 * torch.pi


def shortest_angle_differences(
    *,
    angle_batch: torch.Tensor,  # (...)
) -> torch.Tensor:  # (...)
    """A difference of angles taken the shorter way round: being a full turn
    apart means being in the same place."""
    return (angle_batch + torch.pi) % FULL_TURN - torch.pi


def state_differences(
    *,
    state_batch: torch.Tensor,  # (..., state_dimensionality)
    other_state_batch: torch.Tensor,  # (..., state_dimensionality)
) -> torch.Tensor:  # (..., state_dimensionality)
    """State minus state. Positions subtract, the orientation goes the shorter
    way round, so two states one full turn apart differ by zero rather than by
    2π.

    Every distance and every residual in the pipeline is built from this, which
    is why no stage needs a continuous stand-in for the angle.
    """
    difference_batch = state_batch - other_state_batch
    difference_batch[..., ANGLE_INDEX] = shortest_angle_differences(
        angle_batch=difference_batch[..., ANGLE_INDEX]
    )
    return difference_batch


def advance_states(
    *,
    state_batch: torch.Tensor,  # (..., state_dimensionality)
    state_displacement_batch: torch.Tensor,  # (..., state_dimensionality)
) -> torch.Tensor:  # (..., state_dimensionality)
    """State plus displacement, with the orientation kept inside [0, 2π)."""
    advanced_state_batch = state_batch + state_displacement_batch
    # Not ``%=``: vmap has no batching rule for the in-place remainder and
    # falls back to a loop, which costs 6x here. This runs inside every
    # Jacobian.
    advanced_state_batch[..., ANGLE_INDEX] = (
        advanced_state_batch[..., ANGLE_INDEX] % FULL_TURN
    )
    return advanced_state_batch


def average_states(
    *,
    state_batch: torch.Tensor,  # (..., state_dimensionality)
    dim: int,
) -> torch.Tensor:  # (..., state_dimensionality)
    """The mean state. Positions average arithmetically, the orientation as a
    direction on the unit circle.

    The arithmetic mean of 0.01 and 2π - 0.01 is π — the exact opposite of the
    two orientations it averages. Averaging the unit vectors instead gives the
    orientation they actually surround.
    """
    average_batch = state_batch.mean(dim=dim)
    angle_batch = state_batch[..., ANGLE_INDEX]
    average_batch[..., ANGLE_INDEX] = (
        torch.atan2(
            input=torch.sin(input=angle_batch).mean(dim=dim),
            other=torch.cos(input=angle_batch).mean(dim=dim),
        )
        % FULL_TURN
    )
    return average_batch
