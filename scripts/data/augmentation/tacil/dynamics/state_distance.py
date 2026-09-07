import torch

from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.pusht_state import subtract_states


def calculate_normalized_state_difference(
    *,
    minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    return state_normalizer.normalize_difference(
        difference_batch=subtract_states(
            minuend_batch=minuend_batch, subtrahend_batch=subtrahend_batch
        )
    )


def calculate_normalized_state_distance(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    other_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (batch_size,)
    return torch.linalg.norm(
        calculate_normalized_state_difference(
            minuend_batch=state_batch,
            subtrahend_batch=other_state_batch,
            state_normalizer=state_normalizer,
        ),
        dim=-1,
    )
