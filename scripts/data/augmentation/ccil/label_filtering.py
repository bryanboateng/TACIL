"""Jacobian-weighted label error bound for filtering synthetic labels.

Implements the per-step bound from Deshpande et al. 2024 (Eq. 8): a
synthetic backward step from `head` to `predecessor` under `action` is
risky in proportion to

    ||J_f(head, action)||_2 * ||predecessor - head||

where J_f is the Jacobian of the next-state map. The first factor is how
much the dynamics locally amplify a state perturbation (the local
Lipschitz gain); the second is how far the synthetic step moved. Their
product upper-bounds, to first order, the forward-prediction error the
label could induce.

Everything here is measured in NORMALIZED state space, consistent with
the training loss and the rollout metrics: each dimension contributes in
units of its own std, so a large-scale dimension can't dominate the bound
(and therefore the sigma threshold) just because of its physical units.

The Jacobian of the normalized next-state map equals the raw Jacobian
conjugated by the state-std scaling, J_norm = diag(1/std) J_raw diag(std),
because state and next-state share the same normalization. We compute the
raw Jacobian via autograd and apply that scaling, which only needs the
public `state_std` (the mean is a constant offset and drops out of any
derivative).
"""

import torch

from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel


@torch.no_grad()
def compute_label_error_bound(
    dynamics_model: DynamicsModel,
    anchor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    predecessor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    normalized_next_state_jacobian = _normalized_next_state_jacobian(
        dynamics_model=dynamics_model,
        state_batch=anchor_state_batch,
        action_batch=action_batch,
    )  # (batch_size, state_dimensionality, state_dimensionality)

    spectral_norm = torch.linalg.matrix_norm(
        normalized_next_state_jacobian, ord=2
    )  # (batch_size,)

    normalized_step_distance = torch.linalg.norm(
        dynamics_model.normalize_state(state_batch=predecessor_state_batch)
        - dynamics_model.normalize_state(state_batch=anchor_state_batch),
        dim=-1,
    )  # (batch_size,)

    return spectral_norm * normalized_step_distance


def _normalized_next_state_jacobian(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size, state_dimensionality, state_dimensionality)

    def raw_next_state(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return dynamics_model.predict_next_state(state_batch=state, action_batch=action)

    raw_jacobian = torch.vmap(func=torch.func.jacrev(func=raw_next_state, argnums=0))(
        state_batch, action_batch
    )  # (batch_size, state_dimensionality, state_dimensionality), d next_i / d state_j

    state_std = dynamics_model.state_std  # (state_dimensionality,)

    # J_norm[b, i, j] = (1 / std_i) * J_raw[b, i, j] * std_j
    return raw_jacobian * (state_std[None, None, :] / state_std[None, :, None])
