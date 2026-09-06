from dataclasses import dataclass

import torch

from scripts.data.augmentation.tacil.create_data.state_distance import (
    calculate_state_distances,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)

# Query states compared per chunk. Each one is measured against every recorded
# state at once, so the widest tensor holds chunk x recorded x state, and this
# keeps that inside a few hundred megabytes on the largest budget.
_COMPARISON_CHUNK_SIZE = 256


@dataclass(frozen=True)
class NearestRecordedStateDistances:
    state_distances: torch.Tensor  # (query_count,)
    action_distances: torch.Tensor  # (query_count,)


@torch.no_grad()
def calculate_distances_to_nearest_recorded_state(
    *,
    query_state_batch: torch.Tensor,  # (query_count, state_dimensionality)
    query_action_batch: torch.Tensor,  # (query_count, action_dimensionality)
    recorded_state_batch: torch.Tensor,  # (recorded_count, state_dimensionality)
    recorded_action_batch: torch.Tensor,  # (recorded_count, action_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
    queries_are_the_recorded_states: bool,
) -> NearestRecordedStateDistances:
    state_distance_batches: list[torch.Tensor] = []
    action_distance_batches: list[torch.Tensor] = []

    for chunk_start in range(0, query_state_batch.shape[0], _COMPARISON_CHUNK_SIZE):
        chunk_slice = slice(chunk_start, chunk_start + _COMPARISON_CHUNK_SIZE)
        distance_batch = calculate_state_distances(
            minuend_batch=query_state_batch[chunk_slice].unsqueeze(dim=1),
            subtrahend_batch=recorded_state_batch.unsqueeze(dim=0),
            dynamics_model_ensemble=dynamics_model_ensemble,
        )  # (chunk_size, recorded_count)
        if queries_are_the_recorded_states:
            chunk_indices = torch.arange(
                distance_batch.shape[0], device=distance_batch.device
            )
            distance_batch[chunk_indices, chunk_start + chunk_indices] = float("inf")

        nearest = distance_batch.min(dim=1)
        state_distance_batches.append(nearest.values)
        action_distance_batches.append(
            torch.linalg.norm(
                query_action_batch[chunk_slice]
                - recorded_action_batch[nearest.indices],
                dim=-1,
            )
        )

    return NearestRecordedStateDistances(
        state_distances=torch.cat(tensors=state_distance_batches),
        action_distances=torch.cat(tensors=action_distance_batches),
    )
