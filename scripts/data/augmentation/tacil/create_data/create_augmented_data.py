from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.create_noise_augmentation import compute_noise_std
from scripts.data.augmentation.tacil.create_data.anchor_selection import (
    get_anchor_indices,
)
from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    roll_out_backward,
)
from scripts.data.augmentation.tacil.create_data.config import Config
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    create_augmented_episodes,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    log_dataset_statistics,
    log_rollout_statistics,
)
from scripts.data.augmentation.tacil.create_dynamics.checkpoint import (
    load_dynamics_model_ensemble,
    load_error_model,
)
from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.shared.data import (
    combine_episode_stores,
    load_episode_store,
    upload_episode_store,
)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="create-tacil-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        model_directory_path = Path(
            run.use_artifact(artifact_or_name=config.model_artifact).download()
        )
        dynamics_model_ensemble = load_dynamics_model_ensemble(
            directory_path=model_directory_path, device=device
        )
        dynamics_model_ensemble.eval()
        error_model = load_error_model(directory_path=model_directory_path)
        logger.info(pretty_repr(error_model))
        for field_name, field_value in asdict(error_model).items():
            run.summary[f"error_model/{field_name}"] = field_value

        episode_store = load_episode_store(
            episode_file_path=Path(
                run.use_artifact(artifact_or_name=config.dataset_artifact).download()
            )
        )

        anchor_indices = get_anchor_indices(
            episode_end_indices=episode_store.episode_end_indices,
            expert_part_length=config.expert_part_length,
        )  # (anchor_count,)

        action_noise_std = compute_noise_std(
            samples=episode_store.actions, noise_fraction=config.action_noise_fraction
        )  # (action_dimensionality,)

        #  The median expert step distance is useful as a reference scale.
        #  It answers the question "How far does the expert normally move in one timestep?"
        #  It can be used to judge whether some other quantity (like a predicted or generated step)
        #  is unusually large or small.
        #
        #  We use the median (rather than the mean) because it's robust to
        #  outliers.
        #  A handful of unusually large jumps won't skew the result the way they would an average.
        median_expert_step_distance = calculate_median_step_distance(
            episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
        )

        error_limit = config.error_limit_fraction * median_expert_step_distance

        backward_trajectories, rollout_diagnostics = roll_out_backward(
            dynamics_model_ensemble=dynamics_model_ensemble,
            anchor_state_batch=torch.as_tensor(episode_store.states[anchor_indices]).to(
                device=device
            ),
            anchor_action_batch=torch.as_tensor(
                episode_store.actions[anchor_indices]
            ).to(device=device),
            action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
            error_model=error_model,
            error_limit=error_limit,
            convergence_tolerance=1e-3 * median_expert_step_distance,
        )

        log_rollout_statistics(
            rollout_diagnostics=rollout_diagnostics,
            median_expert_step_distance=median_expert_step_distance,
            error_limit=error_limit,
            run=run,
        )

        augmentation_result = create_augmented_episodes(
            backward_trajectories=backward_trajectories,
            episode_store=episode_store,
            anchor_indices=anchor_indices,
            expert_part_length=config.expert_part_length,
        )

        log_dataset_statistics(
            augmentation_result=augmentation_result,
            source_episode_count=len(episode_store.episode_end_indices),
            source_transition_count=len(episode_store.states),
            run=run,
        )

        upload_episode_store(
            episode_store=combine_episode_stores(
                expert_episode_store=episode_store,
                augmented_episode_store=augmentation_result.episode_store,
            ),
            name="tacil-dataset",
            run=run,
        )


if __name__ == "__main__":
    main()
