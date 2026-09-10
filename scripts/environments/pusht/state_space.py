import torch

AGENT_POSITION_SLICE = slice(0, 2)
BLOCK_POSITION_SLICE = slice(2, 4)
POSITION_SLICE = slice(0, 4)
BLOCK_ORIENTATION_INDEX = 4

# The block orientation is the last entry of a state.
STATE_DIMENSIONALITY = BLOCK_ORIENTATION_INDEX + 1

STATE_GROUPS = (
    ("agent", AGENT_POSITION_SLICE),
    ("block", BLOCK_POSITION_SLICE),
    ("orientation", slice(BLOCK_ORIENTATION_INDEX, BLOCK_ORIENTATION_INDEX + 1)),
)

ARENA_SIZE = 512

FULL_TURN = 2.0 * torch.pi

# The agent's absolute position
_AGENT_POSITION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start

# The orientation of the block as sine and cosine
_BLOCK_ORIENTATION_DIMENSIONALITY = 2

# The block position relative to the agent
_RELATIVE_BLOCK_POSITION_DIMENSIONALITY = (
    BLOCK_POSITION_SLICE.stop - BLOCK_POSITION_SLICE.start
)

# The action relative to the agent
_RELATIVE_ACTION_DIMENSIONALITY = AGENT_POSITION_SLICE.stop - AGENT_POSITION_SLICE.start

NETWORK_INPUT_DIMENSIONALITY = (
    _AGENT_POSITION_DIMENSIONALITY
    + _BLOCK_ORIENTATION_DIMENSIONALITY
    + _RELATIVE_BLOCK_POSITION_DIMENSIONALITY
    + _RELATIVE_ACTION_DIMENSIONALITY
)


class PushTStateSpace:
    """A state is (agent x, agent y, block x, block y, block orientation)."""

    @property
    def dimensionality(self) -> int:
        return STATE_DIMENSIONALITY

    @property
    def network_input_dimensionality(self) -> int:
        return NETWORK_INPUT_DIMENSIONALITY

    def add(
        self,
        *,
        first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        sum_batch = first_addend_batch + second_addend_batch

        # Not `%=`: vmap has no batching rule for the in-place remainder and
        # falls back to a loop, which costs more.
        sum_batch[..., BLOCK_ORIENTATION_INDEX] = (
            sum_batch[..., BLOCK_ORIENTATION_INDEX] % FULL_TURN
        )

        return sum_batch

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        difference_batch = minuend_batch - subtrahend_batch
        difference_batch[..., BLOCK_ORIENTATION_INDEX] = _wrap_angle_to_pi_range(
            angle_batch=difference_batch[..., BLOCK_ORIENTATION_INDEX]
        )
        return difference_batch

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        average_batch = state_batch.mean(dim=0)

        orientation_batch = state_batch[..., BLOCK_ORIENTATION_INDEX]
        average_batch[..., BLOCK_ORIENTATION_INDEX] = (
            torch.atan2(
                input=torch.sin(input=orientation_batch).mean(dim=0),
                other=torch.cos(input=orientation_batch).mean(dim=0),
            )
            % FULL_TURN
        )
        return average_batch

    def create_network_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        block_orientation_batch = state_batch[
            :, BLOCK_ORIENTATION_INDEX : BLOCK_ORIENTATION_INDEX + 1
        ]
        agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]
        return torch.cat(
            tensors=[
                # Absolute agent position
                state_batch[:, AGENT_POSITION_SLICE],
                # Block orientation sine
                torch.sin(block_orientation_batch),
                # Block orientation cosine
                torch.cos(block_orientation_batch),
                # Block position relative to the agent
                state_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch,
                # Action relative to the agent
                action_batch - agent_position_batch,
            ],
            dim=-1,
        )


# ---- Private --------------------------------------------


def _wrap_angle_to_pi_range(
    *,
    angle_batch: torch.Tensor,  # (batch_size,)
) -> torch.Tensor:  # (batch_size,)
    """Wrap angles to [-π, π)."""
    return (angle_batch + torch.pi) % FULL_TURN - torch.pi
