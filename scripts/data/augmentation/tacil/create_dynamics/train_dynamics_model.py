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
from scripts.data.augmentation.tacil.create_dynamics.configure_dashboard import (
    ensure_dashboard_view,
)
from scripts.data.augmentation.tacil.create_dynamics.data import (
    build_windows,
    concatenate_episodes,
    extract_transitions,
)
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    MemberEpochMetrics,
    SplitLipschitz,
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
        ensure_dashboard_view(entity=run.entity, project=run.project)

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
    """Log one epoch's across-member aggregate, live.

    The validation loss and each Lipschitz statistic are logged as an across-member
    mean with a one-standard-deviation band (the standard deviation is the ensemble
    disagreement, and unlike a min/max band a single divergent member cannot blow up
    the y-axis). Each split's local Lipschitz constants are pooled into one histogram so
    the distribution's shape stays visible without one histogram per member. Training
    loss and the worst gradient norm are logged as single health lines. Everything is
    keyed to the epoch, the single timeline now that members advance in lockstep.

    Keys are not prefixed with ``ensemble/``: the whole run is the ensemble, so that
    prefix would put every metric in one group and defeat the natural grouping into
    ``loss/`` and ``lipschitz/`` sections.
    """
    run.log(
        {
            "loss/training_mean": float(
                np.mean([metrics.training_loss for metrics in member_epoch_metrics])
            ),
            "gradient_norm_max": float(
                np.max([metrics.gradient_norm for metrics in member_epoch_metrics])
            ),
            **_scalar_band(
                name="loss/validation",
                member_values=[
                    metrics.validation_loss for metrics in member_epoch_metrics
                ],
            ),
            **_lipschitz_bands(
                split="training",
                member_lipschitz=[
                    metrics.training_lipschitz for metrics in member_epoch_metrics
                ],
            ),
            **_lipschitz_bands(
                split="validation",
                member_lipschitz=[
                    metrics.validation_lipschitz for metrics in member_epoch_metrics
                ],
            ),
        },
        step=epoch_index,
    )


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
    *, split: str, member_lipschitz: list[SplitLipschitz]
) -> dict[str, float | wandb.Histogram]:
    """Mean and p99 bands plus one histogram pooling every member's constants."""
    pooled_local_lipschitz_constants = np.concatenate(
        [split_lipschitz.subsample for split_lipschitz in member_lipschitz]
    )
    return {
        **_scalar_band(
            name=f"lipschitz/mean_{split}",
            member_values=[split_lipschitz.mean for split_lipschitz in member_lipschitz],
        ),
        **_scalar_band(
            name=f"lipschitz/p99_{split}",
            member_values=[split_lipschitz.p99 for split_lipschitz in member_lipschitz],
        ),
        f"lipschitz/distribution_{split}": wandb.Histogram(
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
