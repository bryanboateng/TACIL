import torch

from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.pusht_state import (
    AGENT_POSITION_SLICE,
    BLOCK_ANGLE_INDEX,
    BLOCK_POSITION_SLICE,
)


def network_features(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, NETWORK_FEATURE_DIMENSIONALITY)
    agent_position_batch = state_batch[:, AGENT_POSITION_SLICE]
    return torch.cat(
        tensors=[
            state_batch[:, BLOCK_POSITION_SLICE] - agent_position_batch,
            action_batch - agent_position_batch,
        ],
        dim=-1,
    )


# Block measured from the agent, action measured from the agent.
NETWORK_FEATURE_DIMENSIONALITY = 4
# Those, plus the orientation as a cosine and a sine.
NETWORK_INPUT_DIMENSIONALITY = NETWORK_FEATURE_DIMENSIONALITY + 2


def network_input(
    *, state_batch: torch.Tensor, action_batch: torch.Tensor, normalizers: Normalizers
) -> torch.Tensor:  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
    """Everything the network sees: the scaled features, then the orientation.

    The orientation enters as a point on the unit circle and is left unscaled.
    Two states that show the same block orientation then look the same to the
    network, whether the angle reads 0.3 or 0.3 + 2π, and the circle stays a
    circle.
    """
    angle_batch = state_batch[:, BLOCK_ANGLE_INDEX : BLOCK_ANGLE_INDEX + 1]
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
