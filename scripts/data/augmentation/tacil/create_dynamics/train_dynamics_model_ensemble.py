import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
    MemberEpochMetrics,
    MemberTrainingState,
    TrainingContext,
    create_member_training_state,
    load_best_member_model,
    train_member_for_one_epoch,
)
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
        run.summary["window_count/training"] = train_action_windows.shape[0]
        run.summary["window_count/validation"] = validation_action_windows.shape[0]

        normalizers = Normalizers.from_transitions(
            state_batch=torch.from_numpy(training_transitions.states).to(device=device),
            action_batch=torch.from_numpy(training_transitions.actions).to(
                device=device
            ),
            next_state_batch=torch.from_numpy(training_transitions.next_states).to(
                device=device
            ),
        ).to(device=device)

        training_context = TrainingContext(
            config=config,
            normalizers=normalizers,
            training_transitions=training_transitions,
            validation_transitions=validation_transitions,
            train_state_windows=torch.from_numpy(train_state_windows).to(device=device),
            train_action_windows=torch.from_numpy(train_action_windows).to(
                device=device
            ),
            validation_state_windows=validation_state_windows,
            validation_action_windows=validation_action_windows,
        )

        member_training_states = [
            create_member_training_state(
                member_seed=int(member_seed),
                config=config,
                transitions=training_transitions,
                device=device,
            )
            for member_seed in np.random.SeedSequence(
                entropy=config.seed
            ).generate_state(n_words=config.architecture.ensemble_size)
        ]

        histogram_sample_rng = np.random.default_rng(seed=config.seed)
        for epoch_index in tqdm(
            range(config.training.epoch_count), desc="training ensemble", unit="epoch"
        ):
            _log_epoch(
                epoch_index=epoch_index,
                per_member_metrics=[
                    train_member_for_one_epoch(
                        state=member_training_state,
                        epoch_index=epoch_index,
                        context=training_context,
                    )
                    for member_training_state in member_training_states
                ],
                histogram_sample_rng=histogram_sample_rng,
                run=run,
            )

        _log_checkpoint_summary(member_states=member_training_states, run=run)

        _upload_dynamics_model_ensemble(
            dynamics_model_ensemble=DynamicsModelEnsemble(
                normalizers=normalizers,
                dynamics_models=[
                    load_best_member_model(member_training_state=member_training_state)
                    for member_training_state in member_training_states
                ],
            ),
            run=run,
        )


def _log_epoch(
    *,
    epoch_index: int,
    per_member_metrics: list[MemberEpochMetrics],
    histogram_sample_rng: np.random.Generator,
    run: Run,
) -> None:
    logged_metrics: dict[str, float | wandb.Histogram] = {}
    for member_index, metrics in enumerate(iterable=per_member_metrics):
        member = f"member_{member_index:02d}"
        logged_metrics[f"loss_training/{member}"] = metrics.training_loss
        logged_metrics[f"loss_validation/{member}"] = metrics.validation_loss
        logged_metrics[f"gradient_norm/{member}"] = metrics.gradient_norm

    for split, member_split_lipschitz in (
        (
            "training",
            [
                metrics.training_local_lipschitz_constants
                for metrics in per_member_metrics
            ],
        ),
        (
            "validation",
            [
                metrics.validation_local_lipschitz_constants
                for metrics in per_member_metrics
            ],
        ),
    ):
        # Each member has one Lipschitz constant per transition, which is far more than a
        # histogram needs. Draw a fixed 512 at random from each member and pool them, so
        # the amount logged per epoch stays constant no matter how many transitions exist.
        logged_metrics[f"lipschitz_distribution/{split}"] = wandb.Histogram(
            sequence=np.concatenate(
                [
                    member_lipschitz[
                        histogram_sample_rng.choice(
                            a=member_lipschitz.shape[0], size=512, replace=False
                        )
                    ]
                    for member_lipschitz in member_split_lipschitz
                ]
            ).tolist()
        )

    run.log(logged_metrics, step=epoch_index)


def _log_checkpoint_summary(
    *, member_states: list[MemberTrainingState], run: Run
) -> None:
    checkpoints = []
    for member_state in member_states:
        if member_state.best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")
        checkpoints.append(member_state.best_checkpoint)

    run.summary["checkpoint/validation_loss_mean"] = float(
        np.mean(a=[checkpoint.validation_loss for checkpoint in checkpoints])
    )
    run.summary["checkpoint/epoch_mean"] = float(
        np.mean(a=[checkpoint.epoch for checkpoint in checkpoints])
    )


def _upload_dynamics_model_ensemble(
    *, dynamics_model_ensemble: DynamicsModelEnsemble, run: Run
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        ensemble_directory_path = Path(temporary_directory)
        save_dynamics_model_ensemble(
            dynamics_model_ensemble=dynamics_model_ensemble,
            directory_path=ensemble_directory_path,
        )
        model_artifact = wandb.Artifact(
            name=f"tacil-dynamics-model-{run.name}", type="model"
        )
        model_artifact.add_dir(local_path=str(ensemble_directory_path))
        run.log_artifact(artifact_or_path=model_artifact)


if __name__ == "__main__":
    main()
