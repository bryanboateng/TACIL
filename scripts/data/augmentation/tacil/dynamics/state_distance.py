import torch

from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.pusht_state import subtract_states


def calculate_normalized_state_difference(
    *,
    minuend_batch: torch.Tensor,  # (..., state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (..., state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (..., state_dimensionality)
    """How two states differ, in the normalized state space. That space is what
    makes a displacement of the block comparable to one of the agent, and the
    angle wraps rather than running away."""
    return state_normalizer.normalize_difference(
        difference_batch=subtract_states(
            minuend_batch=minuend_batch, subtrahend_batch=subtrahend_batch
        )
    )


def calculate_state_distances(
    *,
    minuend_batch: torch.Tensor,  # (..., state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (..., state_dimensionality)
    state_normalizer: ZScoreNormalizer,
) -> torch.Tensor:  # (...,)
    return torch.linalg.norm(
        calculate_normalized_state_difference(
            minuend_batch=minuend_batch,
            subtrahend_batch=subtrahend_batch,
            state_normalizer=state_normalizer,
        ),
        dim=-1,
    )
