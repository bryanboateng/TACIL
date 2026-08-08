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

Two more layout facts live here, both unrelated to the periodicity:

- the network reads the agent's position as it is, and the block's position
  relative to the agent (network_features). Contact depends on where the block
  is relative to the agent, so that difference is handed over rather than left
  to be learned. Nothing is lost by it: the absolute block position is the sum
  of the two, so the two coordinate systems carry the same information. Keeping
  the agent's own position is what lets the model see the arena walls at all;
- two states showing the same orientation must look the same to a network, so
  the angle reaches it as a point on the unit circle, never as a number.
"""

import torch

from scripts.data.augmentation.tacil.models.normalizers import Normalizers

# Layout: agent x/y, block x/y, block angle.
STATE_DIMENSIONALITY = 5
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
    state_batch: torch.Tensor,  # (..., STATE_DIMENSIONALITY)
    other_state_batch: torch.Tensor,  # (..., STATE_DIMENSIONALITY)
) -> torch.Tensor:  # (..., STATE_DIMENSIONALITY)
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
    state_batch: torch.Tensor,  # (..., STATE_DIMENSIONALITY)
    state_displacement_batch: torch.Tensor,  # (..., STATE_DIMENSIONALITY)
) -> torch.Tensor:  # (..., STATE_DIMENSIONALITY)
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
    state_batch: torch.Tensor,  # (..., STATE_DIMENSIONALITY)
    dim: int,
) -> torch.Tensor:  # (..., STATE_DIMENSIONALITY)
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


def network_features(
    *,
    state_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, NETWORK_FEATURE_DIMENSIONALITY)
    """The quantities the network reads, before any scaling.

    The agent's position as it is, then the block and the action measured from
    the agent. Contact and the commanded motion both depend on those
    differences, so they are handed over rather than left to be worked out.

    Kept separate from network_input so the scale of each quantity can be
    measured on the quantity itself. Dividing a difference by the spread of an
    absolute position would be the wrong scale.
    """
    agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]
    return torch.cat(
        tensors=[
            agent_position_batch,
            state_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch,
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )


# Agent position, block measured from the agent, action measured from the agent.
NETWORK_FEATURE_DIMENSIONALITY = 6
# Those, plus the orientation as a cosine and a sine.
NETWORK_INPUT_DIMENSIONALITY = NETWORK_FEATURE_DIMENSIONALITY + 2


def network_input(
    *,
    normalizers: Normalizers,
    state_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    """Everything the network sees: the scaled features, then the orientation.

    The orientation enters as a point on the unit circle and is left unscaled.
    Two states that show the same block orientation then look the same to the
    network, whether the angle reads 0.3 or 0.3 + 2π, and the circle stays a
    circle.
    """
    angle_batch = state_batch[:, ANGLE_INDEX : ANGLE_INDEX + 1]
    return torch.cat(
        tensors=[
            normalizers.network_feature_normalizer.normalize(
                value_batch=network_features(
                    state_batch=state_batch, action_batch=action_batch
                )
            ),
            torch.cos(angle_batch),
            torch.sin(angle_batch),
        ],
        dim=-1,
    )
