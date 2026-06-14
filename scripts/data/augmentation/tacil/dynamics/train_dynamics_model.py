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

from scripts.data.augmentation.tacil.dynamics.checkpoint import (
    save_dynamics_model_checkpoint,
)
from scripts.data.augmentation.tacil.dynamics.config import Config
from scripts.data.augmentation.tacil.dynamics.data import (
    build_episodes,
    build_windows,
    concatenate_episodes,
)
from scripts.data.augmentation.tacil.dynamics.evaluation import (
    calculate_local_lipschitz_constants,
    calculate_validation_loss,
)
from scripts.data.augmentation.tacil.dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.dynamics_model import DynamicsModel
from scripts.shared.data import load_episode_dataset
from scripts.shared.logging import create_logger, log_device_info
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
        project="abc",
        job_type="train-tacil-dynamics",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        training_episodes, validation_episodes = train_test_split(
            build_episodes(
                episode_dataset=load_episode_dataset(
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

        train_transitions = concatenate_episodes(episodes=training_episodes)
        validation_transitions = concatenate_episodes(episodes=validation_episodes)

        dynamics_model = DynamicsModel(
            state_dimensionality=train_transitions.states.shape[1],
            action_dimensionality=train_transitions.actions.shape[1],
            hidden_dimensionalities=config.architecture.hidden_layer_sizes,
            lipschitz_constant_upper_bound=config.architecture.lipschitz_constant_upper_bound,
        ).to(device=device)

        dynamics_model.compute_normalization_stats(
            state_batch=torch.from_numpy(train_transitions.states).to(device=device),
            action_batch=torch.from_numpy(train_transitions.actions).to(device=device),
            next_state_batch=torch.from_numpy(train_transitions.next_states).to(
                device=device
            ),
        )

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

        # AveragedModel.module is typed as nn.Module but it is the DynamicsModel we
        # wrapped.
        ema_dynamics_model = cast(DynamicsModel, ema_model.module)

        window_count = train_state_windows_tensor.shape[0]
        steps_per_epoch = max(window_count // config.data.loader.batch_size, 1)

        global_step = 0
        best_checkpoint: Checkpoint | None = None

        for epoch_index in tqdm(
            range(config.training.epoch_count), desc="training", unit="epoch"
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
                    state_window_batch=train_state_windows_tensor[batch_window_indices],
                    action_window_batch=train_action_windows_tensor[
                        batch_window_indices
                    ],
                )
                loss.backward()

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=dynamics_model.parameters(), max_norm=1.0
                )
                optimizer.step()

                run.log(
                    {
                        "loss/training": loss.detach().item(),
                        "gradient_norm": gradient_norm.item(),
                    },
                    step=global_step,
                )
                global_step += 1

            ema_model.update_parameters(model=dynamics_model)

            if config.architecture.lipschitz_constant_upper_bound is not None:
                _refresh_spectral_norm_estimates(ema_dynamics_model=ema_dynamics_model)

            validation_loss = calculate_validation_loss(
                dynamics_model=ema_dynamics_model,
                validation_state_windows=validation_state_windows,
                validation_action_windows=validation_action_windows,
                batch_size=config.data.loader.batch_size,
            )
            run.log({"loss/validation": validation_loss}, step=global_step)
            run.log(
                {
                    **_compute_lipschitz_metrics(
                        dynamics_model=ema_dynamics_model,
                        states=train_transitions.states,
                        actions=train_transitions.actions,
                        batch_size=config.data.loader.batch_size,
                        split="training",
                    ),
                    **_compute_lipschitz_metrics(
                        dynamics_model=ema_dynamics_model,
                        states=validation_transitions.states,
                        actions=validation_transitions.actions,
                        batch_size=config.data.loader.batch_size,
                        split="validation",
                    ),
                },
                step=global_step,
            )
            run.log({"epoch": epoch_index}, step=global_step)

            if (
                best_checkpoint is None
                or validation_loss < best_checkpoint.validation_loss
            ):
                best_checkpoint = Checkpoint(
                    validation_loss=validation_loss,
                    epoch=epoch_index,
                    state_dict=copy.deepcopy(ema_dynamics_model.state_dict()),
                )

        if best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")

        dynamics_model.load_state_dict(state_dict=best_checkpoint.state_dict)
        run.summary["selected/epoch"] = best_checkpoint.epoch
        run.summary["selected/validation_loss"] = best_checkpoint.validation_loss

        _upload_model(dynamics_model=dynamics_model, run=run)


def _compute_lipschitz_metrics(
    *,
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
    split: str,
) -> dict[str, float | wandb.Histogram]:
    local_lipschitz_constants = calculate_local_lipschitz_constants(
        dynamics_model=dynamics_model,
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
    """Recompute the spectral-norm vectors for the EMA model's averaged weights.

    These vectors only update during train-mode forward passes, and the EMA model is
    never trained, so its vectors are out of date. They also can't be copied from the
    live model, because they only fit the exact weights they were computed from, and
    the EMA model has different weights. So we run a few forward
    passes in train mode: each pass moves the vectors one step closer to the correct
    values for the current weights.
    The input values play no role here, so we use zeros.
    """
    model_device = next(ema_dynamics_model.parameters()).device

    dummy_state_batch = torch.zeros(
        1, ema_dynamics_model.state_dimensionality, device=model_device
    )
    dummy_action_batch = torch.zeros(
        1, ema_dynamics_model.action_dimensionality, device=model_device
    )

    model_was_training = ema_dynamics_model.training
    ema_dynamics_model.train()
    for _ in range(50):
        ema_dynamics_model.predict_next_state(
            state_batch=dummy_state_batch, action_batch=dummy_action_batch
        )
    ema_dynamics_model.train(mode=model_was_training)


def _upload_model(*, dynamics_model: DynamicsModel, run: Run) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        dynamics_checkpoint_path = Path(temporary_directory) / "dynamics.pt"
        save_dynamics_model_checkpoint(
            dynamics_model=dynamics_model, path=dynamics_checkpoint_path
        )
        model_artifact = wandb.Artifact(
            name=f"tacil-dynamics-model-{run.name}", type="model"
        )
        model_artifact.add_file(
            local_path=str(dynamics_checkpoint_path), name="dynamics.pt"
        )
        run.log_artifact(model_artifact)


if __name__ == "__main__":
    main()
