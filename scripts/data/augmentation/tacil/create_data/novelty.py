"""How much of the augmented set is ground the recorded set does not cover.

Counting invented transitions says how many were made, never whether they are
worth making. A chain that stays on the demonstration it came from produces
states the policy already has, under labels it already has, and teaches it
nothing; a chain that leaves the demonstration is the only kind that can teach
recovery. The difference is a distance, so it can simply be measured.

Distances are reported in multiples of the median expert step, which is the
same yardstick the chain error limit uses. Below one, a synthetic state sits
closer to a recorded state than two recorded states sit to each other, and the
augmentation is a copy of the data with extra steps.
"""

import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import subtract_states

# Synthetic states compared per chunk, so that the widest tensor holds
# chunk x recorded x state.
NOVELTY_CHUNK_SIZE = 256


@torch.no_grad()
def measure_distances_to_recorded_states(
    *,
    synthetic_state_batch: torch.Tensor,  # (synthetic_count, state_dimensionality)
    recorded_state_batch: torch.Tensor,  # (recorded_count, state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (synthetic_count,)
    """For each synthetic state, the distance to the nearest recorded one."""
    nearest_distance_chunks: list[torch.Tensor] = []

    for chunk_start in range(0, synthetic_state_batch.shape[0], NOVELTY_CHUNK_SIZE):
        chunk_state_batch = synthetic_state_batch[
            chunk_start : chunk_start + NOVELTY_CHUNK_SIZE
        ]  # (chunk_size, state_dimensionality)

        distance_batch = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=chunk_state_batch.unsqueeze(dim=1),
                    subtrahend_batch=recorded_state_batch.unsqueeze(dim=0),
                )
            ),
            dim=-1,
        )  # (chunk_size, recorded_count)

        nearest_distance_chunks.append(distance_batch.min(dim=1).values)

    return torch.cat(tensors=nearest_distance_chunks)
