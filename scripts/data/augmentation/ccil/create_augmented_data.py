from dataclasses import dataclass

import numpy as np
import torch
from wandb import Run

from scripts.data.augmentation.anchor_selection import get_anchor_indices
from scripts.data.augmentation.assembly import assemble_episodes
from scripts.data.augmentation.ccil.config import (
    BackwardEulerLabels,
    CcilConfig,
    ForwardEulerLabels,
    LabelConfig,
    NoisyActionLabels,
    RejectionScore,
)
from scripts.data.augmentation.ccil.world_model import WorldModel, fit_world_model
from scripts.shared import stage_names
from scripts.shared.distribution_plot import create_distribution_plot
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episodes import Episodes, calculate_indices_with_successor
from scripts.shared.log import create_logger
from scripts.shared.model_artifact import DYNAMICS_MODEL_ARTIFACT_TYPE, upload_model
from scripts.shared.seeding import set_seed

logger = create_logger()

STAGE_NAME = stage_names.DATA
DYNAMICS_STAGE_NAME = stage_names.DYNAMICS

EPOCH_METRIC_NAME = f"{DYNAMICS_STAGE_NAME}/0_epoch"

LOSS_SECTION = f"{DYNAMICS_STAGE_NAME}/1_loss"

_BATCH_SIZE = 4_096


def create_ccil_augmented_data(
    *,
    config: CcilConfig,
    expert_episodes: Episodes,
    expert_part_length: int,
    seed: int,
    device: torch.device,
    run: Run,
) -> Episodes:
    set_seed(seed=seed)

    states, actions, next_states = (
        torch.from_numpy(x).to(device=device)
        for x in _extract_transitions(episodes=expert_episodes)
    )

    world_model = WorldModel(
        state_dimensionality=states.shape[1],
        action_dimensionality=actions.shape[1],
        hidden_layer_sizes=config.dynamics.hidden_layer_sizes,
        lipschitz_type=config.dynamics.lipschitz_type,
        lipschitz_constraint=config.dynamics.lipschitz_constraint,
    ).to(device=device)

    epoch_losses = fit_world_model(
        world_model=world_model,
        config=config.dynamics,
        state_batch=states,
        action_batch=actions,
        next_state_batch=next_states,
    )
    _define_epoch_metrics(run=run)
    for epoch_index, losses in enumerate(iterable=epoch_losses):
        run.log(
            data={
                f"{LOSS_SECTION}/1_total": losses.total,
                f"{LOSS_SECTION}/2_data_term": losses.data_term,
                f"{LOSS_SECTION}/3_regularization_term": losses.regularization_term,
                EPOCH_METRIC_NAME: epoch_index,
            }
        )
    if config.dynamics.save_model:
        upload_model(
            model=world_model, artifact_type=DYNAMICS_MODEL_ARTIFACT_TYPE, run=run
        )

    anchor_indices = np.intersect1d(
        get_anchor_indices(
            episode_end_indices=expert_episodes.episode_end_indices,
            expert_part_length=expert_part_length,
        ),
        calculate_indices_with_successor(
            episode_end_indices=expert_episodes.episode_end_indices,
            total_timestep_count=expert_episodes.states.shape[0],
        ),
    )  # (anchor_count,)

    with torch.no_grad():
        labels = _generate_labels(
            world_model=world_model,
            state_batch=torch.from_numpy(expert_episodes.states[anchor_indices]).to(
                device=device
            ),
            action_batch=torch.from_numpy(expert_episodes.actions[anchor_indices]).to(
                device=device
            ),
            next_state_batch=torch.from_numpy(
                expert_episodes.states[anchor_indices + 1]
            ).to(device=device),
            anchor_index_batch=torch.from_numpy(anchor_indices).to(device=device),
            config=config,
        )

    accepted_mask = _accept_labels(
        label_scores=labels.rejection_scores, config=config.labels
    )
    label_count = labels.states.shape[0]
    accepted_count = int(accepted_mask.sum())
    rejected_count = label_count - accepted_count

    run.summary[f"{STAGE_NAME}/1_labels/1_expert_transition_count"] = states.shape[0]
    run.summary[f"{STAGE_NAME}/1_labels/2_label_count"] = label_count
    run.summary[f"{STAGE_NAME}/1_labels/4_rejected_label_count"] = rejected_count
    run.summary[f"{STAGE_NAME}/1_labels/6_unsolved_label_count"] = (
        labels.attempted_count - label_count
    )
    run.summary[f"{STAGE_NAME}/1_labels/5_rejected_label_fraction"] = (
        rejected_count / max(label_count, 1)
    )

    for label, value in summarize_distribution(values=labels.distances).items():
        run.summary[f"{STAGE_NAME}/2_label_distances/{label}"] = value
    run.log(
        data={
            f"{STAGE_NAME}/2_label_distances/4_plot": create_distribution_plot(
                values=labels.distances, xlabel="label distance"
            )
        }
    )

    for label, value in summarize_distribution(
        values=labels.local_lipschitz_estimates
    ).items():
        run.summary[f"{STAGE_NAME}/4_local_lipschitz_estimates/{label}"] = value
    for label, value in summarize_distribution(values=labels.rejection_scores).items():
        run.summary[f"{STAGE_NAME}/5_rejection_scores/{label}"] = value

    run.summary[f"{STAGE_NAME}/3_dataset/1_synthetic_transition_count"] = accepted_count
    if accepted_count == 0:
        raise RuntimeError(
            "Rejection removed every CCIL label, so there is nothing to augment with."
        )

    return assemble_episodes(
        expert_episodes=expert_episodes,
        synthetic_states=labels.states[accepted_mask],
        synthetic_actions=labels.actions[accepted_mask],
        trajectory_lengths=np.ones(accepted_count, dtype=np.int64),
        anchor_indices=labels.anchor_indices[accepted_mask],
        expert_part_length=expert_part_length,
    )


def _define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    run.define_metric(name=f"{LOSS_SECTION}/*", step_metric=EPOCH_METRIC_NAME)


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _Labels:
    states: np.ndarray  # (label_count, state_dimensionality)
    actions: np.ndarray  # (label_count, action_dimensionality)
    anchor_indices: np.ndarray  # (label_count,)
    distances: np.ndarray  # (label_count,)
    local_lipschitz_estimates: np.ndarray  # (label_count,)
    rejection_scores: np.ndarray  # (label_count,)
    attempted_count: int


def _accept_labels(
    *, label_scores: np.ndarray, config: LabelConfig
) -> np.ndarray:  # (label_count,)
    rejection = config.rejection
    match rejection.rule:
        case "none":
            return np.ones_like(label_scores, dtype=bool)
        case "quantile":
            limit = float(np.quantile(a=label_scores, q=rejection.limit))
        case "absolute":
            limit = rejection.limit

    return label_scores < limit


def _calculate_rejection_scores(
    *,
    distances: np.ndarray,  # (label_count,)
    local_lipschitz_estimates: np.ndarray,  # (label_count,)
    score: RejectionScore,
) -> np.ndarray:  # (label_count,)
    match score:
        case "distance":
            return distances
        case "jacobian_weighted_distance":
            return local_lipschitz_estimates * distances


def _estimate_local_lipschitz_constants(
    *,
    world_model: WorldModel,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    state_dimensionality = state_batch.shape[-1]

    def predict_residual(state_action: torch.Tensor) -> torch.Tensor:
        return world_model.predict_residual(
            state_batch=state_action[None, :state_dimensionality],
            action_batch=state_action[None, state_dimensionality:],
        )[0]

    with torch.enable_grad():
        jacobian_batch = torch.func.vmap(
            torch.func.jacrev(predict_residual)
        )(
            torch.cat(tensors=[state_batch, action_batch], dim=-1)
        )  # (batch_size, state_dimensionality, state_dimensionality + action_dimensionality)
    return torch.linalg.matrix_norm(jacobian_batch.detach(), ord=2)


def _extract_transitions(
    *, episodes: Episodes
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices_with_successor = calculate_indices_with_successor(
        episode_end_indices=episodes.episode_end_indices,
        total_timestep_count=episodes.states.shape[0],
    )
    return (
        episodes.states[indices_with_successor],
        episodes.actions[indices_with_successor],
        episodes.states[indices_with_successor + 1],
    )


def _generate_labels(
    *,
    world_model: WorldModel,
    state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    next_state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    anchor_index_batch: torch.Tensor,  # (anchor_count,)
    config: CcilConfig,
) -> _Labels:
    generation = config.labels.generation
    label_count_per_transition = (
        generation.label_count_per_transition
        if isinstance(generation, NoisyActionLabels)
        else 1
    )

    label_state_chunks: list[np.ndarray] = []
    label_action_chunks: list[np.ndarray] = []
    anchor_index_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    lipschitz_chunks: list[np.ndarray] = []
    attempted_count = 0

    for batch_start_index in range(0, state_batch.shape[0], _BATCH_SIZE):
        batch_slice = slice(batch_start_index, batch_start_index + _BATCH_SIZE)
        repeated_states = state_batch[batch_slice].repeat_interleave(
            repeats=label_count_per_transition, dim=0
        )
        repeated_actions = action_batch[batch_slice].repeat_interleave(
            repeats=label_count_per_transition, dim=0
        )
        repeated_anchor_indices = anchor_index_batch[batch_slice].repeat_interleave(
            repeats=label_count_per_transition, dim=0
        )

        match generation:
            case NoisyActionLabels(action_noise_std=action_noise_std):
                repeated_next_states = next_state_batch[batch_slice].repeat_interleave(
                    repeats=label_count_per_transition, dim=0
                )
                label_actions = repeated_actions + torch.normal(
                    mean=0.0,
                    std=action_noise_std,
                    size=repeated_actions.shape,
                    device=repeated_actions.device,
                )
                label_states = repeated_next_states - world_model.predict_residual(
                    state_batch=repeated_next_states, action_batch=label_actions
                )
                solved_mask = _every_label(reference_batch=label_states)
            case BackwardEulerLabels():
                label_actions = repeated_actions
                label_states = repeated_states - world_model.predict_residual(
                    state_batch=repeated_states, action_batch=label_actions
                )
                solved_mask = _every_label(reference_batch=label_states)
            case ForwardEulerLabels(max_iteration_count=max_iteration_count):
                label_actions = repeated_actions
                label_states, solved_mask = _solve_forward_euler(
                    world_model=world_model,
                    target_state_batch=repeated_states,
                    action_batch=label_actions,
                    max_iteration_count=max_iteration_count,
                )

        distances = torch.linalg.norm(label_states - repeated_states, dim=-1)
        local_lipschitz_estimates = _estimate_local_lipschitz_constants(
            world_model=world_model,
            state_batch=state_batch[batch_slice],
            action_batch=action_batch[batch_slice],
        ).repeat_interleave(repeats=label_count_per_transition, dim=0)

        attempted_count += label_states.shape[0]
        label_state_chunks.append(label_states[solved_mask].cpu().numpy())
        label_action_chunks.append(label_actions[solved_mask].cpu().numpy())
        anchor_index_chunks.append(repeated_anchor_indices[solved_mask].cpu().numpy())
        distance_chunks.append(distances[solved_mask].cpu().numpy())
        lipschitz_chunks.append(local_lipschitz_estimates[solved_mask].cpu().numpy())

    distances = np.concatenate(distance_chunks)
    local_lipschitz_estimates = np.concatenate(lipschitz_chunks)
    return _Labels(
        states=np.concatenate(label_state_chunks).astype(np.float32),
        actions=np.concatenate(label_action_chunks).astype(np.float32),
        anchor_indices=np.concatenate(anchor_index_chunks).astype(np.int64),
        distances=distances,
        local_lipschitz_estimates=local_lipschitz_estimates,
        rejection_scores=_calculate_rejection_scores(
            distances=distances,
            local_lipschitz_estimates=local_lipschitz_estimates,
            score=config.labels.rejection.score,
        ),
        attempted_count=attempted_count,
    )


def _every_label(
    *,
    reference_batch: torch.Tensor,  # (label_count, state_dimensionality)
) -> torch.Tensor:  # (label_count,)
    return torch.ones(
        reference_batch.shape[0], dtype=torch.bool, device=reference_batch.device
    )


def _solve_forward_euler(
    *,
    world_model: WorldModel,
    target_state_batch: torch.Tensor,  # (label_count, state_dimensionality)
    action_batch: torch.Tensor,  # (label_count, action_dimensionality)
    max_iteration_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # states, and which ones were solved
    threshold = 1e-4

    def calculate_residual(
        state: torch.Tensor, action: torch.Tensor, target_state: torch.Tensor
    ) -> torch.Tensor:
        return (
            state
            + world_model.predict_residual(state_batch=state, action_batch=action)
            - target_state
        )

    calculate_jacobians = torch.func.vmap(
        torch.func.jacrev(calculate_residual, argnums=0)
    )

    state_batch = target_state_batch.clone()
    solved_mask = torch.zeros(
        state_batch.shape[0], dtype=torch.bool, device=state_batch.device
    )
    searching_mask = ~solved_mask

    with torch.enable_grad():
        for _ in range(max_iteration_count):
            with torch.no_grad():
                residual_batch = calculate_residual(
                    state_batch, action_batch, target_state_batch
                )

            newly_solved_mask = (
                torch.linalg.norm(residual_batch, dim=-1) <= threshold
            ) & searching_mask
            solved_mask |= newly_solved_mask
            searching_mask &= ~newly_solved_mask
            if not bool(searching_mask.any()):
                break

            searching_indices = searching_mask.nonzero(as_tuple=True)[0]
            step_batch = torch.linalg.solve(
                calculate_jacobians(
                    state_batch[searching_indices],
                    action_batch[searching_indices],
                    target_state_batch[searching_indices],
                ).detach(),
                residual_batch[searching_indices],
            )
            state_batch[searching_indices] = state_batch[searching_indices] - step_batch

    return state_batch, solved_mask
