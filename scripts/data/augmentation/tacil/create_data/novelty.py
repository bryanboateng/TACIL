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

from scripts.data.augmentation.tacil.create_data.state_distance import (
    iterate_distances_to_recorded_states,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


@torch.no_grad()
def measure_distances_to_recorded_states(
    *,
    synthetic_state_batch: torch.Tensor,  # (synthetic_count, state_dimensionality)
    recorded_state_batch: torch.Tensor,  # (recorded_count, state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (synthetic_count,)
    """For each synthetic state, the distance to the nearest recorded one."""
    return torch.cat(
        tensors=[
            distance_batch.min(dim=1).values
            for _, distance_batch in iterate_distances_to_recorded_states(
                query_state_batch=synthetic_state_batch,
                recorded_state_batch=recorded_state_batch,
                dynamics_model_ensemble=dynamics_model_ensemble,
            )
        ]
    )
