import copy
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro
import wandb
from numpy import typing as npt
from rich.pretty import pretty_repr
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from wandb import Run

from scripts.data.augmentation.tacil.create_dynamics.checkpoint import (
    save_dynamics_model_ensemble,
)
from scripts.data.augmentation.tacil.create_dynamics.config import Config
from scripts.data.augmentation.tacil.create_dynamics.data import (
    TransitionSet,
    build_windows,
    concatenate_episodes,
    extract_transitions,
)
from scripts.data.augmentation.tacil.create_dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    calculate_validation_loss,
)
from scripts.data.augmentation.tacil.create_dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.mmmmmmm.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.mmmmmmm.dynamics_network import DynamicsModel
from scripts.data.augmentation.tacil.mmmmmmm.normalizers import Normalizers
from scripts.shared.data import load_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed

logger = create_logger()


@dataclass
class Checkpoint:
    validation_loss: float
    epoch: int
    state_dict: dict


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    set_seed(seed=config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="train-tacil-dynamics",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        training_episodes, validation_episodes = train_test_split(
            extract_transitions(
                episode_store=load_episode_store(
                    episode_file_path=Path(
                        run.use_artifact(
                            artifact_or_name=config.data.artifact
                        ).download()
                    )
                )
            ),
            test_size=config.data.split.validation_fraction,
            random_state=config.data.split.seed,
            shuffle=True,
        )

        training_transitions = concatenate_episodes(episodes=training_episodes)
        validation_transitions = concatenate_episodes(episodes=validation_episodes)

        train_state_windows, train_action_windows = build_windows(
            episodes=training_episodes, window_length=config.data.loader.window_length
        )
        validation_state_windows, validation_action_windows = build_windows(
            episodes=validation_episodes, window_length=config.data.loader.window_length
        )
        logger.info(
            "Built %d training windows of length %d.",
            train_action_windows.shape[0],
            config.data.loader.window_length,
        )

        train_state_windows_tensor = torch.from_numpy(train_state_windows).to(
            device=device
        )
        train_action_windows_tensor = torch.from_numpy(train_action_windows).to(
            device=device
        )

        normalizers = Normalizers.from_transitions(
            state_batch=torch.from_numpy(training_transitions.states).to(device=device),
            action_batch=torch.from_numpy(training_transitions.actions).to(
                device=device
            ),
            next_state_batch=torch.from_numpy(training_transitions.next_states).to(
                device=device
            ),
        ).to(device=device)

        dynamics_models: list[DynamicsModel] = []
        global_step = 0
        for member_index, member_seed in enumerate(
            np.random.SeedSequence(entropy=config.seed).generate_state(
                n_words=config.architecture.ensemble_size
            )
        ):
            set_seed(seed=int(member_seed))

            dynamics_model, global_step = _train_single_member(
                member_index=member_index,
                config=config,
                normalizers=normalizers,
                training_transitions=training_transitions,
                validation_transitions=validation_transitions,
                train_state_windows_tensor=train_state_windows_tensor,
                train_action_windows_tensor=train_action_windows_tensor,
                validation_state_windows=validation_state_windows,
                validation_action_windows=validation_action_windows,
                run=run,
                device=device,
                initial_global_step=global_step,
            )
            dynamics_models.append(dynamics_model)

        _upload_dynamics_model_ensemble(
            dynamics_model_ensemble=DynamicsModelEnsemble(
                normalizers=normalizers, dynamics_models=dynamics_models
            ),
            run=run,
        )


def _train_single_member(
    *,
    member_index: int,
    config: Config,
    normalizers: Normalizers,
    training_transitions: TransitionSet,
    validation_transitions: TransitionSet,
    train_state_windows_tensor: torch.Tensor,
    train_action_windows_tensor: torch.Tensor,
    validation_state_windows: npt.NDArray[np.float32],
    validation_action_windows: npt.NDArray[np.float32],
    run: Run,
    device: torch.device,
    initial_global_step: int,
) -> tuple[DynamicsModel, int]:
    metric_prefix = f"member_{member_index}/"

    dynamics_model = DynamicsModel(
        state_dimensionality=training_transitions.states.shape[1],
        action_dimensionality=training_transitions.actions.shape[1],
        hidden_dimensionalities=config.architecture.hidden_layer_sizes,
        lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
    ).to(device=device)

    optimizer = torch.optim.Adam(
        params=dynamics_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    ema_model = AveragedModel(
        dynamics_model,
        avg_fn=lambda ema_parameter, current_parameter, _num_averaged: (
            config.training.ema_decay * ema_parameter
            + (1.0 - config.training.ema_decay) * current_parameter
        ),
        use_buffers=False,
    )

    # AveragedModel.module is typed as nn.Module but it is the DynamicsNetwork we
    # wrapped.
    ema_dynamics_model = cast(DynamicsModel, ema_model.module)

    window_count = train_state_windows_tensor.shape[0]
    steps_per_epoch = max(window_count // config.data.loader.batch_size, 1)

    global_step = initial_global_step
    best_checkpoint: Checkpoint | None = None

    for epoch_index in tqdm(
        range(config.training.epoch_count),
        desc=f"training member {member_index}",
        unit="epoch",
    ):
        dynamics_model.train()

        shuffled_window_indices = torch.randperm(n=window_count, device=device)
        for step_index in range(steps_per_epoch):
            batch_window_indices = shuffled_window_indices[
                step_index * config.data.loader.batch_size : (step_index + 1)
                * config.data.loader.batch_size
            ]

            optimizer.zero_grad()

            loss = compute_loss(
                dynamics_model=dynamics_model,
                normalizers=normalizers,
                state_window_batch=train_state_windows_tensor[batch_window_indices],
                action_window_batch=train_action_windows_tensor[batch_window_indices],
            )
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters=dynamics_model.parameters(), max_norm=1.0
            )
            optimizer.step()

            run.log(
                {
                    f"{metric_prefix}loss/training": loss.detach().item(),
                    f"{metric_prefix}gradient_norm": gradient_norm.item(),
                },
                step=global_step,
            )
            global_step += 1

        ema_model.update_parameters(model=dynamics_model)

        if config.architecture.lipschitz_constant_upper_bound is not None:
            _refresh_spectral_norm_estimates(ema_dynamics_model=ema_dynamics_model)

        validation_loss = calculate_validation_loss(
            dynamics_model=ema_dynamics_model,
            normalizers=normalizers,
            state_windows=validation_state_windows,
            action_windows=validation_action_windows,
            batch_size=config.data.loader.batch_size,
        )
        run.log({f"{metric_prefix}loss/validation": validation_loss}, step=global_step)
        run.log(
            {
                f"{metric_prefix}{key}": value
                for key, value in {
                    **_compute_lipschitz_metrics(
                        dynamics_model=ema_dynamics_model,
                        normalizers=normalizers,
                        states=training_transitions.states,
                        actions=training_transitions.actions,
                        batch_size=config.data.loader.batch_size,
                        split="training",
                    ),
                    **_compute_lipschitz_metrics(
                        dynamics_model=ema_dynamics_model,
                        normalizers=normalizers,
                        states=validation_transitions.states,
                        actions=validation_transitions.actions,
                        batch_size=config.data.loader.batch_size,
                        split="validation",
                    ),
                }.items()
            },
            step=global_step,
        )
        run.log({f"{metric_prefix}epoch": epoch_index}, step=global_step)

        if best_checkpoint is None or validation_loss < best_checkpoint.validation_loss:
            best_checkpoint = Checkpoint(
                validation_loss=validation_loss,
                epoch=epoch_index,
                state_dict=copy.deepcopy(ema_dynamics_model.state_dict()),
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    dynamics_model.load_state_dict(state_dict=best_checkpoint.state_dict)
    run.summary[f"{metric_prefix}selected/epoch"] = best_checkpoint.epoch
    run.summary[f"{metric_prefix}selected/validation_loss"] = (
        best_checkpoint.validation_loss
    )

    return dynamics_model, global_step


def _compute_lipschitz_metrics(
    *,
    dynamics_model: DynamicsModel,
    normalizers: Normalizers,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
    split: str,
) -> dict[str, float | wandb.Histogram]:
    local_lipschitz_constants = calculate_local_lipschitz_constants(
        dynamics_model=dynamics_model,
        normalizers=normalizers,
        states=states,
        actions=actions,
        batch_size=batch_size,
    )
    return {
        f"lipschitz/distribution_{split}": wandb.Histogram(
            local_lipschitz_constants.tolist()
        ),
        f"lipschitz/mean_{split}": float(np.mean(local_lipschitz_constants)),
        f"lipschitz/p99_{split}": float(
            np.percentile(local_lipschitz_constants, q=99.0)
        ),
    }


@torch.no_grad()
def _refresh_spectral_norm_estimates(*, ema_dynamics_model: DynamicsModel) -> None:
    """Recompute the spectral-norm vectors for the EMA network's averaged weights.

    These vectors only update during train-mode forward passes, and the EMA network is
    never trained, so its vectors are out of date. They also can't be copied from the
    live network, because they only fit the exact weights they were computed from, and
    the EMA network has different weights. So we run a few forward
    passes in train mode: each pass moves the vectors one step closer to the correct
    values for the current weights.
    The input values play no role here, so we use zeros.
    """
    network_device = next(ema_dynamics_model.parameters()).device

    dummy_normalized_input_batch = torch.zeros(
        1,
        ema_dynamics_model.state_dimensionality
        + ema_dynamics_model.action_dimensionality,
        device=network_device,
    )

    model_was_training = ema_dynamics_model.training
    ema_dynamics_model.train()
    for _ in range(50):
        ema_dynamics_model(dummy_normalized_input_batch)
    ema_dynamics_model.train(mode=model_was_training)


def _upload_dynamics_model_ensemble(
    *, dynamics_model_ensemble: DynamicsModelEnsemble, run: Run
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        ensemble_directory = Path(temporary_directory)
        save_dynamics_model_ensemble(
            dynamics_model_ensemble=dynamics_model_ensemble,
            directory=ensemble_directory,
        )
        model_artifact = wandb.Artifact(
            name=f"tacil-dynamics-model-{run.name}", type="model"
        )
        model_artifact.add_dir(local_path=str(ensemble_directory))
        run.log_artifact(model_artifact)


if __name__ == "__main__":
    main()
