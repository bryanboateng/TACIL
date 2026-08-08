"""What the numbers in a PushT state mean.

Everywhere else a state is just a vector: the dynamics model, the backward
rollout and the error account never look inside it. This module is the one
place that does — which dimension is a position, which one is an orientation,
and what follows from that:

- an orientation is periodic, so a recorded full turn is a jump of -2π that no
  model can predict (unwrap_angles removes it, wrap_angles puts labels back
  into the range the policy uses);
- contact depends on where things are relative to the agent, not on where in
  the arena that happens (measure_positions_from_agent in network_input);
- two states showing the same orientation should look the same to a network
  (the angle enters as a point on the unit circle).
"""

import numpy as np
import torch

from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.data import EpisodeStore

# Frame layout: agent x/y, block x/y, block angle. A dynamics state is one
# such frame, or two when the previous state is stacked in.
FRAME_SIZE = 5
AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
ANGLE_INDEX = 4

# The parts a state error is broken down into.
STATE_GROUPS = (
    ("agent", AGENT_POSITION_SLICE),
    ("block", BLOCK_POSITION_SLICE),
    ("angle", slice(ANGLE_INDEX, ANGLE_INDEX + 1)),
)

# The quadratic arena (gym_pusht window size, pixels).
ARENA_SIZE = 512

FULL_TURN = 2.0 * np.pi


def unwrap_angles(*, episode_store: EpisodeStore) -> EpisodeStore:
    """Undo the [0, 2π) wrapping of the block angle inside every episode.

    A wrap is a -2π step that no model can predict: it happens wherever the
    block passes one fixed orientation, and it dwarfs the real rotation per
    step — in the PushT data it inflates the typical angular step by 12x.
    """
    states = episode_store.states.copy()
    episode_start_indices = np.concatenate(
        [[0], episode_store.episode_end_indices[:-1]]
    )

    for start_index, end_index in zip(
        episode_start_indices, episode_store.episode_end_indices
    ):
        states[start_index:end_index, ANGLE_INDEX] = np.unwrap(
            p=states[start_index:end_index, ANGLE_INDEX]
        )

    return EpisodeStore(
        states=states,
        actions=episode_store.actions,
        episode_end_indices=episode_store.episode_end_indices,
    )


def wrap_angles(
    *,
    states: np.ndarray,  # (..., state_dimensionality)
) -> np.ndarray:  # (..., state_dimensionality)
    """Put the block angle back into [0, 2π), the range the policy and the
    environment use."""
    wrapped_states = states.copy()
    wrapped_states[..., ANGLE_INDEX] %= FULL_TURN
    return wrapped_states


def shortest_angle_differences(
    *,
    angle_batch: torch.Tensor,  # (...)
) -> torch.Tensor:  # (...)
    """A difference of angles taken the shorter way round: being a full turn
    apart means being in the same place."""
    return (angle_batch + torch.pi) % FULL_TURN - torch.pi


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
