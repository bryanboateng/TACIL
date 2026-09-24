import numpy as np
import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import (
    BackwardEulerLabels,
    CcilConfig,
    ForwardEulerLabels,
    LabelConfig,
    NoisyActionLabels,
)
from scripts.data.augmentation.ccil.world_model import WorldModel, fit_world_model
from scripts.shared import stage_names
from scripts.shared.distribution_plot import create_distribution_plot
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episodes import Episodes
from scripts.shared.log import create_logger
from scripts.shared.seeding import set_seed
from scripts.shared.state_action_transitions import StateActionTransitions

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
    seed: int,
    device: torch.device,
    run: Run,
) -> StateActionTransitions:
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

    with torch.no_grad():
        label_states, label_actions, label_distances, attempted_count = (
            _generate_labels(
                world_model=world_model,
                state_batch=states,
                action_batch=actions,
                next_state_batch=next_states,
                config=config,
            )
        )

    accepted_mask = _accept_labels(
        label_distances=label_distances, config=config.labels
    )
    label_count = label_states.shape[0]
    rejected_count = label_count - int(accepted_mask.sum())

    run.summary[f"{STAGE_NAME}/1_labels/1_expert_transition_count"] = states.shape[0]
    run.summary[f"{STAGE_NAME}/1_labels/2_label_count"] = label_count
    run.summary[f"{STAGE_NAME}/1_labels/4_rejected_label_count"] = rejected_count
    run.summary[f"{STAGE_NAME}/1_labels/6_unsolved_label_count"] = (
        attempted_count - label_count
    )
    run.summary[f"{STAGE_NAME}/1_labels/5_rejected_label_fraction"] = (
        rejected_count / max(label_count, 1)
    )

    for label, value in summarize_distribution(values=label_distances).items():
        run.summary[f"{STAGE_NAME}/2_label_distances/{label}"] = value
    run.log(
        data={
            f"{STAGE_NAME}/2_label_distances/4_plot": create_distribution_plot(
                values=label_distances, xlabel="label distance"
            )
        }
    )

    accepted_states = label_states[accepted_mask]
    run.summary[f"{STAGE_NAME}/3_dataset/1_synthetic_transition_count"] = int(
        accepted_states.shape[0]
    )
    return StateActionTransitions(
        states=accepted_states, actions=label_actions[accepted_mask]
    )


def _define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    run.define_metric(name=f"{LOSS_SECTION}/*", step_metric=EPOCH_METRIC_NAME)


# ---- Private --------------------------------------------


def _accept_labels(
    *, label_distances: np.ndarray, config: LabelConfig
) -> np.ndarray:  # (label_count,)
    rejection = config.rejection
    match rejection.rule:
        case "none":
            return np.ones_like(label_distances, dtype=bool)
        case "quantile":
            limit = float(np.quantile(a=label_distances, q=rejection.limit))
        case "absolute":
            limit = rejection.limit

    return label_distances < limit


def _extract_transitions(
    *, episodes: Episodes
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_timestep_indices = episodes.episode_end_indices - 1
    keep_mask = np.ones(episodes.states.shape[0], dtype=bool)
    keep_mask[last_timestep_indices] = False

    successor_indices = np.nonzero(keep_mask)[0] + 1
    return (
        episodes.states[keep_mask],
        episodes.actions[keep_mask],
        episodes.states[successor_indices],
    )


def _generate_labels(
    *,
    world_model: WorldModel,
    state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
    next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    config: CcilConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    generation = config.labels.generation
    label_count_per_transition = (
        generation.label_count_per_transition
        if isinstance(generation, NoisyActionLabels)
        else 1
    )

    label_state_chunks: list[np.ndarray] = []
    label_action_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    attempted_count = 0

    for batch_start_index in range(0, state_batch.shape[0], _BATCH_SIZE):
        batch_slice = slice(batch_start_index, batch_start_index + _BATCH_SIZE)
        repeated_states = state_batch[batch_slice].repeat_interleave(
            repeats=label_count_per_transition, dim=0
        )
        repeated_actions = action_batch[batch_slice].repeat_interleave(
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

        attempted_count += label_states.shape[0]
        label_state_chunks.append(label_states[solved_mask].cpu().numpy())
        label_action_chunks.append(label_actions[solved_mask].cpu().numpy())
        distance_chunks.append(distances[solved_mask].cpu().numpy())

    return (
        np.concatenate(label_state_chunks).astype(np.float32),
        np.concatenate(label_action_chunks).astype(np.float32),
        np.concatenate(distance_chunks),
        attempted_count,
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
                ),
                residual_batch[searching_indices],
            )
            state_batch[searching_indices] = state_batch[searching_indices] - step_batch

    return state_batch, solved_mask
