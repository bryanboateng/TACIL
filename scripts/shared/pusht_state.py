"""What the numbers in a PushT state mean.

A PushT state is four positions and one orientation. The orientation is
periodic, so the state is not a plain vector: 0.01 and 2π - 0.01 are almost the
same orientation, but they are far apart as numbers. Only this one fact makes
the module necessary. Everything below follows from it.

The standard remedy is to leave the state alone and to replace the three
arithmetic operations that would otherwise be wrong:

    a - b        ->  state_differences(a, b)
    a + d        ->  advance_states(a, d)
    x.mean(dim)  ->  average_states(x, dim)

Every stage then keeps working on flat vectors. Hertzberg et al. call the first
two operators ⊟ and ⊞ and name exactly this pattern; see "Integrating generic
sensor fusion algorithms with sound state representations through encapsulation
of manifolds", Information Fusion 14(1), 2013, arXiv:1107.1119. average_states
is the matching circular mean from directional statistics.

Two more layout facts live here, both unrelated to the periodicity:

- contact depends on where things are relative to the agent, not on where in
  the arena that happens (measure_positions_from_agent in network_input);
- two states showing the same orientation must look the same to a network, so
  the angle reaches it as a point on the unit circle, never as a number.
"""

import torch

from scripts.data.augmentation.tacil.models.normalizers import Normalizers

# Frame layout: agent x/y, block x/y, block angle. A dynamics state is one
# such frame, or two when the previous state is stacked in.
FRAME_SIZE = 5
AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
ANGLE_INDEX = 4
# Every frame carries an angle, so this picks all of them at once.
ANGLE_SLICE = slice(ANGLE_INDEX, None, FRAME_SIZE)

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
    """State minus state. Positions subtract, orientations go the shorter way
    round, so two states one full turn apart differ by zero rather than by 2π.

    Every distance and every residual in the pipeline is built from this, which
    is why no stage needs a continuous stand-in for the angle.
    """
    difference_batch = state_batch - other_state_batch
    difference_batch[..., ANGLE_SLICE] = shortest_angle_differences(
        angle_batch=difference_batch[..., ANGLE_SLICE]
    )
    return difference_batch


def advance_states(
    *,
    state_batch: torch.Tensor,  # (..., state_dimensionality)
    state_displacement_batch: torch.Tensor,  # (..., state_dimensionality)
) -> torch.Tensor:  # (..., state_dimensionality)
    """State plus displacement, with orientations kept inside [0, 2π)."""
    advanced_state_batch = state_batch + state_displacement_batch
    # Not ``%=``: vmap has no batching rule for the in-place remainder and
    # falls back to a loop, which costs 6x here. This runs inside every
    # Jacobian.
    advanced_state_batch[..., ANGLE_SLICE] = (
        advanced_state_batch[..., ANGLE_SLICE] % FULL_TURN
    )
    return advanced_state_batch


def average_states(
    *,
    state_batch: torch.Tensor,  # (..., state_dimensionality)
    dim: int,
) -> torch.Tensor:  # (..., state_dimensionality)
    """The mean state. Positions average arithmetically, orientations as
    directions on the unit circle.

    The arithmetic mean of 0.01 and 2π - 0.01 is π — the exact opposite of the
    two orientations it averages. Averaging the unit vectors instead gives the
    orientation they actually surround.
    """
    average_batch = state_batch.mean(dim=dim)
    angle_batch = state_batch[..., ANGLE_SLICE]
    average_batch[..., ANGLE_SLICE] = (
        torch.atan2(
            input=torch.sin(input=angle_batch).mean(dim=dim),
            other=torch.cos(input=angle_batch).mean(dim=dim),
        )
        % FULL_TURN
    )
    return average_batch


def network_input_dimensionality(
    *,
    state_dimensionality: int,
    action_dimensionality: int,
    measure_positions_from_agent: bool,
) -> int:
    frame_count = state_dimensionality // FRAME_SIZE
    # Each frame's angle turns into a cosine and a sine: one number more.
    dimensionality = state_dimensionality + frame_count + action_dimensionality
    # Measured from the agent, the agent's own position is always zero.
    return dimensionality - 2 if measure_positions_from_agent else dimensionality


def network_input(
    *,
    normalizers: Normalizers,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    measure_positions_from_agent: bool,
) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
    """Everything the network sees.

    With measure_positions_from_agent, positions are measured from the agent
    instead of from the arena corner: contact depends on where things are
    relative to the agent, not on where in the arena that happens, so the model
    learns the contact rule once instead of once per place.

    The orientation is always a point on the unit circle. Two states that show
    the same block orientation then look the same to the network, whether the
    angle reads 0.3 or 0.3 + 2π.
    """
    agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]

    parts = [
        (action_batch - agent_position_batch) / normalizers.action_normalizer.std
        if measure_positions_from_agent
        else normalizers.action_normalizer.normalize(value_batch=action_batch)
    ]

    for frame_start in range(0, state_batch.shape[-1], FRAME_SIZE):
        frame_batch = state_batch[:, frame_start : frame_start + FRAME_SIZE]
        frame_shift = normalizers.state_normalizer.mean[
            frame_start : frame_start + FRAME_SIZE
        ]
        frame_scale = normalizers.state_normalizer.std[
            frame_start : frame_start + FRAME_SIZE
        ]

        if measure_positions_from_agent:
            if frame_start > 0:
                # The current agent position is the origin, so only earlier
                # frames still carry one.
                parts.append(
                    (frame_batch[:, AGENT_POSITION_SLICE] - agent_position_batch)
                    / frame_scale[AGENT_POSITION_SLICE]
                )
            parts.append(
                (frame_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch)
                / frame_scale[BLOCK_POSITION_SLICE]
            )
        else:
            positions_slice = slice(
                AGENT_POSITION_SLICE.start, BLOCK_POSITION_SLICE.stop
            )
            parts.append(
                (frame_batch[:, positions_slice] - frame_shift[positions_slice])
                / frame_scale[positions_slice]
            )

        angle_batch = frame_batch[:, ANGLE_INDEX : ANGLE_INDEX + 1]
        parts.append(
            torch.cat(tensors=[torch.cos(angle_batch), torch.sin(angle_batch)], dim=-1)
        )

    return torch.cat(tensors=parts, dim=-1)
