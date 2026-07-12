import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from sklearn.model_selection import train_test_split
from wandb import Run

from scripts.data.augmentation.tacil.create_dynamics.checkpoint import (
    save_dynamics_model_ensemble,
)
from scripts.data.augmentation.tacil.create_dynamics.config import Config
from scripts.data.augmentation.tacil.create_dynamics.data import (
    build_windows,
    concatenate_episodes,
    extract_transitions,
)
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    LocalLipschitzEpochHistory,
    MemberEpochHistory,
    train_dynamics_model,
)
from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.data import load_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    set_seed(seed=config.seed)

    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="train-tacil-dynamics",
        config=asdict(obj=config),
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
        member_epoch_histories: list[MemberEpochHistory] = []
        global_step = 0
        for member_index, member_seed in enumerate(
            np.random.SeedSequence(entropy=config.seed).generate_state(
                n_words=config.architecture.ensemble_size
            )
        ):
            set_seed(seed=int(member_seed))

            dynamics_model, global_step, member_epoch_history = train_dynamics_model(
                member_index=member_index,
                config=config,
                normalizers=normalizers,
                training_transitions=training_transitions,
                validation_transitions=validation_transitions,
                train_state_windows=train_state_windows,
                train_action_windows=train_action_windows,
                validation_state_windows=validation_state_windows,
                validation_action_windows=validation_action_windows,
                run=run,
                device=device,
                initial_global_step=global_step,
            )
            dynamics_models.append(dynamics_model)
            member_epoch_histories.append(member_epoch_history)

        _log_ensemble_aggregate_metrics(
            run=run,
            member_epoch_histories=member_epoch_histories,
            initial_global_step=global_step,
        )

        _upload_dynamics_model_ensemble(
            dynamics_model_ensemble=DynamicsModelEnsemble(
                normalizers=normalizers, dynamics_models=dynamics_models
            ),
            run=run,
        )


def _log_ensemble_aggregate_metrics(
    *,
    run: Run,
    member_epoch_histories: list[MemberEpochHistory],
    initial_global_step: int,
) -> None:
    """Log the ensemble's per-epoch scalar bands and pooled Lipschitz histograms.

    For each scalar metric this emits the across-member mean and the two band edges
    (mean minus and plus one standard deviation), so a line panel renders the mean
    bracketed by a disagreement band. The standard deviation (a quantity TACIL cares
    about) is used rather than a min/max band so a single divergent member cannot blow
    up the y-axis. For each split, every member's local Lipschitz constants are pooled
    into a single histogram per epoch, so the shape of the ensemble's Lipschitz
    distribution stays visible without one histogram per member. All series are plotted
    against an epoch x-axis (``ensemble/epoch``) rather than the global step, which the
    per-step training-health curves consume sequentially.
    """
    run.define_metric(name="ensemble/epoch")
    run.define_metric(name="ensemble/*", step_metric="ensemble/epoch")

    epoch_count = len(member_epoch_histories[0].validation_losses)

    global_step = initial_global_step
    for epoch_index in range(epoch_count):
        aggregate_metrics: dict[str, float | wandb.Histogram] = {
            "ensemble/epoch": epoch_index,
            **_scalar_band(
                name="ensemble/loss/validation",
                member_values=[
                    history.validation_losses[epoch_index]
                    for history in member_epoch_histories
                ],
            ),
            **_lipschitz_bands(
                split="training",
                member_lipschitz_histories=[
                    history.training_lipschitz for history in member_epoch_histories
                ],
                epoch_index=epoch_index,
            ),
            **_lipschitz_bands(
                split="validation",
                member_lipschitz_histories=[
                    history.validation_lipschitz for history in member_epoch_histories
                ],
                epoch_index=epoch_index,
            ),
        }
        run.log(aggregate_metrics, step=global_step)
        global_step += 1


def _scalar_band(*, name: str, member_values: list[float]) -> dict[str, float]:
    """Across-member mean and the two edges of a one-standard-deviation band."""
    member_mean = float(np.mean(member_values))
    member_standard_deviation = float(np.std(member_values))
    return {
        f"{name}_mean": member_mean,
        f"{name}_band_lower": member_mean - member_standard_deviation,
        f"{name}_band_upper": member_mean + member_standard_deviation,
    }


def _lipschitz_bands(
    *,
    split: str,
    member_lipschitz_histories: list[LocalLipschitzEpochHistory],
    epoch_index: int,
) -> dict[str, float | wandb.Histogram]:
    """Mean and p99 bands plus one histogram pooling every member's constants."""
    pooled_local_lipschitz_constants = np.concatenate(
        [
            lipschitz_history.subsamples[epoch_index]
            for lipschitz_history in member_lipschitz_histories
        ]
    )
    return {
        **_scalar_band(
            name=f"ensemble/lipschitz/mean_{split}",
            member_values=[
                lipschitz_history.means[epoch_index]
                for lipschitz_history in member_lipschitz_histories
            ],
        ),
        **_scalar_band(
            name=f"ensemble/lipschitz/p99_{split}",
            member_values=[
                lipschitz_history.p99s[epoch_index]
                for lipschitz_history in member_lipschitz_histories
            ],
        ),
        f"ensemble/lipschitz/distribution_{split}": wandb.Histogram(
            pooled_local_lipschitz_constants.tolist()
        ),
    }


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
