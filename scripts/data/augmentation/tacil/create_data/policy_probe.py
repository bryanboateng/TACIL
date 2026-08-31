"""Finds out where a learned policy drifts, using only the data and the model.

TACIL anchors on recorded steps, so the corrective tube it builds is centred on
the expert's path. A policy fails somewhere else: it leaves the path, and the
states it then occupies are the ones it has never been told what to do in.

The probe closes that gap without asking the simulator anything. A small
behaviour-cloning policy is fitted to the data at hand, and then run forward
inside the dynamics ensemble from disturbed recorded states. Where it ends up
is a data-only estimate of the region a trained policy will actually occupy.

The probe is not the policy that will be trained later, and on a diffusion
policy run it is not even the same policy class. It is a stand-in, and its job
is only to point at a region -- the chains that grow there are still built and
still checked the same way.
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.policies.bc.policy import BehaviorCloningPolicyModel
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.transition_batch import TransitionBatch

PROBE_BATCH_SIZE = 512
PROBE_HIDDEN_LAYER_DIMENSIONALITY = 256
PROBE_LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class ProbeDiagnostics:
    start_count: int
    step_count: int
    final_training_loss: float
    # How far the probe drifts from where it started, in normalized units. Near
    # zero would mean the probe stands still and its states say nothing about
    # where a policy goes.
    median_drift: float


@torch.no_grad()
def roll_out_probe_policy(
    *,
    probe_policy: BehaviorCloningPolicyModel,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    start_state_batch: torch.Tensor,  # (start_count, state_dimensionality)
    rollout_length: int,
) -> torch.Tensor:  # (start_count * rollout_length, state_dimensionality)
    probe_policy.eval()

    state_batch = start_state_batch.clone()
    visited_state_batches: list[torch.Tensor] = []

    for _ in tqdm(
        iterable=range(rollout_length), desc="probe step", unit="step", leave=False
    ):
        state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=state_batch,
            action_batch=probe_policy.predict_action(observation_batch=state_batch),
        )
        visited_state_batches.append(state_batch)

    return torch.cat(tensors=visited_state_batches, dim=0)


def train_probe_policy(
    *, transitions: TransitionBatch, epoch_count: int, seed: int, device: torch.device
) -> tuple[BehaviorCloningPolicyModel, float]:
    """Fits the stand-in policy. No validation, because validating it would
    mean rolling episodes in the simulator, and the method never does that."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    state_batch = torch.as_tensor(data=transitions.states).to(device=device)
    action_batch = torch.as_tensor(data=transitions.actions).to(device=device)

    probe_policy = BehaviorCloningPolicyModel(
        observation_dimensionality=state_batch.shape[1],
        action_dimensionality=action_batch.shape[1],
        hidden_layer_dimensionality=PROBE_HIDDEN_LAYER_DIMENSIONALITY,
        observation_normalizer=ZScoreNormalizer(value_batch=state_batch),
        action_normalizer=MinMaxNormalizer(value_batch=action_batch),
    ).to(device=device)

    optimizer = torch.optim.Adam(
        params=probe_policy.parameters(), lr=PROBE_LEARNING_RATE
    )
    loss_function = torch.nn.MSELoss()

    normalized_action_batch = probe_policy.normalize_action(action_batch=action_batch)
    transition_count = state_batch.shape[0]

    probe_policy.train()
    epoch_loss = float("nan")
    for _ in tqdm(
        iterable=range(epoch_count), desc="probe epoch", unit="epoch", leave=False
    ):
        shuffled_indices = torch.randperm(n=transition_count, generator=generator).to(
            device=device
        )

        losses: list[torch.Tensor] = []
        for batch_start in range(0, transition_count, PROBE_BATCH_SIZE):
            batch_indices = shuffled_indices[
                batch_start : batch_start + PROBE_BATCH_SIZE
            ]

            optimizer.zero_grad()
            loss = loss_function(
                probe_policy.network(
                    probe_policy.normalize_observation(
                        observation_batch=state_batch[batch_indices]
                    )
                ),
                normalized_action_batch[batch_indices],
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())

        epoch_loss = float(torch.stack(tensors=losses).mean())

    return probe_policy, epoch_loss
