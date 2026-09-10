import torch

from scripts.environments.state_space import StateSpace
from scripts.shared.normalization import ZScoreNormalizer


def calculate_normalized_state_difference(
    *,
    minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    state_space: StateSpace,
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    return state_normalizer.normalize_difference(
        difference_batch=state_space.subtract(
            minuend_batch=minuend_batch, subtrahend_batch=subtrahend_batch
        )
    )


def calculate_normalized_state_distance(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    other_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    state_space: StateSpace,
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size,)
    return torch.linalg.norm(
        calculate_normalized_state_difference(
            minuend_batch=state_batch,
            subtrahend_batch=other_state_batch,
            state_space=state_space,
            state_normalizer=state_normalizer,
        ),
        dim=-1,
    )
