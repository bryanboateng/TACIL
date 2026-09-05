from collections.abc import Iterator

import numpy as np
import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import subtract_states

# Query states compared per chunk. Each one is measured against every recorded
# state at once, so the widest tensor holds chunk x recorded x state, and this
# keeps that inside a few hundred megabytes on the largest budget.
COMPARISON_CHUNK_SIZE = 256


def calculate_state_distances(
    *,
    minuend_batch: torch.Tensor,  # (..., state_dimensionality)
    subtrahend_batch: torch.Tensor,  # (..., state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (...,)
    """How far apart two states lie, in the normalized state space. That space
    is what makes a displacement of the block comparable to one of the agent,
    and the angle wraps rather than running away."""
    return torch.linalg.norm(
        dynamics_model_ensemble.state_normalizer.normalize_difference(
            difference_batch=subtract_states(
                minuend_batch=minuend_batch, subtrahend_batch=subtrahend_batch
            )
        ),
        dim=-1,
    )


def iterate_distances_to_recorded_states(
    *,
    query_state_batch: torch.Tensor,  # (query_count, state_dimensionality)
    recorded_state_batch: torch.Tensor,  # (recorded_count, state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> Iterator[tuple[int, torch.Tensor]]:  # chunk start, (chunk, recorded_count)
    """Every query's distance to every recorded state, a chunk of queries at a
    time. The whole cross product does not fit at once, and what a caller keeps
    from a chunk -- the nearest, the nearest few -- differs."""
    for chunk_start in range(0, query_state_batch.shape[0], COMPARISON_CHUNK_SIZE):
        yield (
            chunk_start,
            calculate_state_distances(
                minuend_batch=query_state_batch[
                    chunk_start : chunk_start + COMPARISON_CHUNK_SIZE
                ].unsqueeze(dim=1),
                subtrahend_batch=recorded_state_batch.unsqueeze(dim=0),
                dynamics_model_ensemble=dynamics_model_ensemble,
            ),
        )


@torch.no_grad()
def measure_step_distance(
    *, episode_store: EpisodeStore, dynamics_model_ensemble: DynamicsModelEnsemble
) -> float:
    device = next(dynamics_model_ensemble.parameters()).device

    state_batch = torch.as_tensor(data=episode_store.states).to(
        device=device
    )  # (total_timestep_count, state_dimensionality)

    # For every consecutive pair of states (s_t, s_{t+1}), compute how far
    # apart they are in space.
    step_distance_batch = calculate_state_distances(
        minuend_batch=state_batch[1:],
        subtrahend_batch=state_batch[:-1],
        dynamics_model_ensemble=dynamics_model_ensemble,
    )  # (total_timestep_count - 1,)

    # Because all episodes inside the episode store are concatenated into
    # one long tensor, some "consecutive" pairs are not actually consecutive
    # in time.
    # They are the last state of one episode paired with the first state of
    # the next episode.
    # We need to exclude those pairs.
    episode_boundary_mask = np.zeros(shape=len(step_distance_batch), dtype=bool)
    episode_boundary_mask[episode_store.episode_end_indices[:-1] - 1] = True

    return float(
        step_distance_batch[
            ~torch.as_tensor(data=episode_boundary_mask).to(device=device)
        ]
        .median()
        .item()
    )
