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

        context = TrainingContext(
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

        # Train every member in lockstep: all advance one epoch before the next epoch
        # begins, so the across-member aggregate can be logged live each epoch.
        member_states = [
            create_member_training_state(
                member_index=member_index,
                member_seed=int(member_seed),
                config=config,
                training_transitions=training_transitions,
                device=device,
            )
            for member_index, member_seed in enumerate(
                np.random.SeedSequence(entropy=config.seed).generate_state(
                    n_words=config.architecture.ensemble_size
                )
            )
        ]

        for epoch_index in tqdm(
            range(config.training.epoch_count),
            desc="training ensemble",
            unit="epoch",
        ):
            _log_ensemble_epoch(
                run=run,
                epoch_index=epoch_index,
                member_epoch_metrics=[
                    train_member_for_one_epoch(
                        state=state, epoch_index=epoch_index, context=context
                    )
                    for state in member_states
                ],
            )

        _upload_dynamics_model_ensemble(
            dynamics_model_ensemble=DynamicsModelEnsemble(
                normalizers=normalizers,
                dynamics_models=[
                    load_best_member_model(state=state) for state in member_states
                ],
            ),
            run=run,
        )


def _log_ensemble_epoch(
    *,
    run: Run,
    epoch_index: int,
    member_epoch_metrics: list[MemberEpochMetrics],
) -> None:
    """Log every member's raw per-epoch metrics, plus the pooled Lipschitz histograms.

    Nothing is aggregated in code: each member's value is logged under a
    ``<quantity>/member_XX`` key. In W&B, selecting all of a quantity's member keys on a
    panel's Y axis and enabling metric grouping (Agg = Mean, Range = Std Dev) draws a
    mean + standard-deviation band. Keeping the raw per-member values in the one run means
    the band, the individual members, and their identity are all recoverable from the
    same run. The two Lipschitz distribution histograms pool a bounded subsample of every
    member's constants to show the distribution's shape. Everything is keyed to the epoch,
    the single timeline now that members advance in lockstep.
    """
    logged_metrics: dict[str, float | wandb.Histogram] = {}
    for member_index, metrics in enumerate(member_epoch_metrics):
        member = f"member_{member_index:02d}"
        logged_metrics[f"loss_training/{member}"] = metrics.training_loss
        logged_metrics[f"loss_validation/{member}"] = metrics.validation_loss
        logged_metrics[f"gradient_norm/{member}"] = metrics.gradient_norm

    for split, member_split_lipschitz in (
        ("training", [metrics.training_lipschitz for metrics in member_epoch_metrics]),
        (
            "validation",
            [metrics.validation_lipschitz for metrics in member_epoch_metrics],
        ),
    ):
        logged_metrics[f"lipschitz_distribution/{split}"] = wandb.Histogram(
            np.concatenate(member_split_lipschitz).tolist()
        )

    run.log(logged_metrics, step=epoch_index)


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
