import numpy as np
import torch
from numpy import typing as npt
from tqdm import tqdm
from wandb import Run

from scripts.policies.mopo.config import DynamicsConfig
from scripts.policies.mopo.dynamics import (
    Dynamics,
    DynamicsModelEnsemble,
    InputNormalizer,
    TerminationFunction,
)
from scripts.policies.mopo.transitions import Transitions
from scripts.shared import stage_names

STAGE_NAME = stage_names.DYNAMICS

EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"
TRAINING_SECTION = f"{STAGE_NAME}/1_training"
VALIDATION_SECTION = f"{STAGE_NAME}/2_validation"


def train_dynamics(
    *,
    config: DynamicsConfig,
    rollout_penalty_coefficient: float,
    transitions: Transitions,
    termination_function: TerminationFunction,
    seed: int,
    device: torch.device,
    run: Run,
) -> Dynamics:
    _define_epoch_metrics(run=run)

    generator = np.random.default_rng(seed=seed)

    input_batch = np.concatenate(
        [transitions.states, transitions.actions], axis=-1
    )  # (transition_count, input_dimensionality)
    target_batch = np.concatenate(
        [transitions.next_states - transitions.states, transitions.rewards], axis=-1
    )  # (transition_count, target_dimensionality)

    transition_count = len(input_batch)
    holdout_count = min(int(transition_count * config.holdout_ratio), 1000)
    shuffled_indices = generator.permutation(x=transition_count)
    holdout_indices = shuffled_indices[:holdout_count]
    training_indices = shuffled_indices[holdout_count:]

    input_normalizer = InputNormalizer(value_batch=input_batch[training_indices])

    training_input_batch = input_normalizer.normalize(
        value_batch=input_batch[training_indices]
    )
    training_target_batch = target_batch[training_indices]
    holdout_input_batch = torch.as_tensor(
        data=input_normalizer.normalize(value_batch=input_batch[holdout_indices]),
        device=device,
    )
    holdout_target_batch = torch.as_tensor(
        data=target_batch[holdout_indices], device=device
    )

    model_ensemble = DynamicsModelEnsemble(
        config=config,
        state_dimensionality=transitions.states.shape[1],
        action_dimensionality=transitions.actions.shape[1],
    ).to(device=device)
    optimizer = torch.optim.Adam(
        params=model_ensemble.parameters(), lr=config.learning_rate
    )

    run.summary[f"{STAGE_NAME}/0_dataset/1_training_transition_count"] = len(
        training_indices
    )
    run.summary[f"{STAGE_NAME}/0_dataset/2_holdout_transition_count"] = holdout_count

    member_indices = generator.integers(
        low=0,
        high=len(training_indices),
        size=(config.ensemble_size, len(training_indices)),
    )

    best_holdout_losses = [1e10] * config.ensemble_size
    epoch_count_without_improvement = 0
    epoch_index = 0

    progress_bar = tqdm(desc="Dynamics epoch", unit="epoch")
    while epoch_count_without_improvement < config.patience_epoch_count:
        training_loss = _train_for_one_epoch(
            config=config,
            model_ensemble=model_ensemble,
            optimizer=optimizer,
            input_batch=training_input_batch[member_indices],
            target_batch=training_target_batch[member_indices],
            device=device,
        )
        holdout_losses = _calculate_holdout_losses(
            model_ensemble=model_ensemble,
            input_batch=holdout_input_batch,
            target_batch=holdout_target_batch,
        )

        improved_member_indices = [
            index
            for index, (holdout_loss, best_holdout_loss) in enumerate(
                zip(holdout_losses, best_holdout_losses, strict=True)
            )
            if (best_holdout_loss - holdout_loss) / best_holdout_loss > 0.01
        ]
        for index in improved_member_indices:
            best_holdout_losses[index] = holdout_losses[index]

        if improved_member_indices:
            model_ensemble.save_members(member_indices=improved_member_indices)
            epoch_count_without_improvement = 0
        else:
            epoch_count_without_improvement += 1

        run.log(
            data={
                f"{TRAINING_SECTION}/1_loss": training_loss,
                f"{VALIDATION_SECTION}/1_elite_loss": _calculate_elite_loss(
                    holdout_losses=holdout_losses, elite_count=config.elite_count
                ),
                EPOCH_METRIC_NAME: epoch_index,
            }
        )

        member_indices = _shuffle_rows(index_batch=member_indices, generator=generator)
        epoch_index += 1
        progress_bar.update()
    progress_bar.close()

    elite_indices = sorted(
        range(config.ensemble_size), key=lambda index: best_holdout_losses[index]
    )[: config.elite_count]
    model_ensemble.set_elites(member_indices=elite_indices)
    model_ensemble.restore_members()
    model_ensemble.eval()

    run.summary[f"{STAGE_NAME}/3_result/1_epoch_count"] = epoch_index
    run.summary[f"{STAGE_NAME}/3_result/2_elite_loss"] = _calculate_elite_loss(
        holdout_losses=best_holdout_losses, elite_count=config.elite_count
    )

    return Dynamics(
        model_ensemble=model_ensemble,
        optimizer=optimizer,
        input_normalizer=input_normalizer,
        termination_function=termination_function,
        penalty_coefficient=rollout_penalty_coefficient,
        device=device,
    )


# ---- Private --------------------------------------------


def _train_for_one_epoch(
    *,
    config: DynamicsConfig,
    model_ensemble: DynamicsModelEnsemble,
    optimizer: torch.optim.Optimizer,
    input_batch: npt.NDArray[np.float32],  # (ensemble_size, sample_count, input_dim)
    target_batch: npt.NDArray[np.float32],  # (ensemble_size, sample_count, target_dim)
    device: torch.device,
) -> float:
    model_ensemble.train()

    sample_count = input_batch.shape[1]
    losses: list[float] = []

    for start_index in range(0, sample_count, config.batch_size):
        end_index = start_index + config.batch_size

        mean_batch, log_variance_batch = model_ensemble(
            torch.as_tensor(data=input_batch[:, start_index:end_index], device=device)
        )  # each (ensemble_size, batch_size, target_dimensionality)

        inverse_variance_batch = torch.exp(input=-log_variance_batch)
        target_tensor = torch.as_tensor(
            data=target_batch[:, start_index:end_index], device=device
        )

        # Gaussian negative log likelihood, averaged over batch and dimension
        # and summed over the ensemble.
        loss = (
            ((mean_batch - target_tensor).square() * inverse_variance_batch)
            .mean(dim=(1, 2))
            .sum()
            + log_variance_batch.mean(dim=(1, 2)).sum()
            + model_ensemble.calculate_decay_loss()
            + config.log_variance_loss_coefficient
            * (
                model_ensemble.maximum_log_variance.sum()
                - model_ensemble.minimum_log_variance.sum()
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def _calculate_holdout_losses(
    *,
    model_ensemble: DynamicsModelEnsemble,
    input_batch: torch.Tensor,  # (holdout_count, input_dimensionality)
    target_batch: torch.Tensor,  # (holdout_count, target_dimensionality)
) -> list[float]:
    model_ensemble.eval()
    with torch.no_grad():
        mean_batch, _ = model_ensemble(input_batch)
        return (mean_batch - target_batch).square().mean(dim=(1, 2)).tolist()


def _calculate_elite_loss(*, holdout_losses: list[float], elite_count: int) -> float:
    return float(np.mean(np.sort(holdout_losses)[:elite_count]))


def _shuffle_rows(
    *,
    index_batch: npt.NDArray[np.int64],  # (ensemble_size, sample_count)
    generator: np.random.Generator,
) -> npt.NDArray[np.int64]:  # (ensemble_size, sample_count)
    order = np.argsort(generator.uniform(size=index_batch.shape), axis=-1)
    return index_batch[np.arange(index_batch.shape[0])[:, None], order]


def _define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    for name in (f"{TRAINING_SECTION}/*", f"{VALIDATION_SECTION}/*"):
        run.define_metric(name=name, step_metric=EPOCH_METRIC_NAME)
