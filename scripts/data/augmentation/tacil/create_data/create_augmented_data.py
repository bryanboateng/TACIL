from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.create_noise_augmentation import compute_noise_std
from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    roll_out_backward,
)
from scripts.data.augmentation.tacil.create_data.config import Config
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    combine_episode_datasets,
    create_augmented_episodes,
    get_anchor_indices,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    log_dataset_statistics,
)
from scripts.data.augmentation.tacil.dynamics.checkpoint import load_dynamics_model
from scripts.shared.data import load_episode_dataset, upload_episode_dataset
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
        project="abc",
        job_type="create-tacil-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        dynamics_model = load_dynamics_model(
            checkpoint_file_path=Path(
                run.use_artifact(artifact_or_name=config.model_artifact).download()
            )
            / "dynamics.pt",
            device=device,
        )
        dynamics_model.eval()

        episode_dataset = load_episode_dataset(
            episode_file_path=Path(
                run.use_artifact(artifact_or_name=config.dataset_artifact).download()
            )
        )

        anchor_indices = get_anchor_indices(
            episode_end_indices=episode_dataset.episode_end_indices,
            expert_part_length=config.expert_part_length,
        )  # (anchor_count,)

        action_noise_std = compute_noise_std(
            samples=episode_dataset.actions, noise_scale=config.action_noise_scale
        )  # (action_dimensionality,)

        backward_trajectories = roll_out_backward(
            dynamics_model=dynamics_model,
            state_batch=torch.as_tensor(episode_dataset.states[anchor_indices]).to(
                device=device
            ),
            action_batch=torch.as_tensor(episode_dataset.actions[anchor_indices]).to(
                device=device
            ),
            step_count=config.synthetic_part_length,
            action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
            max_newton_iteration_count=config.maximum_newton_iterations,
        )

        augmentation_result = create_augmented_episodes(
            backward_trajectories=backward_trajectories,
            episode_dataset=episode_dataset,
            anchor_indices=anchor_indices,
            expert_part_length=config.expert_part_length,
            label_error_bound_quantile=config.label_error_bound_quantile,
        )

        log_dataset_statistics(
            augmentation_result=augmentation_result,
            source_episode_count=len(episode_dataset.episode_end_indices),
            source_transition_count=len(episode_dataset.states),
            run=run,
        )

        upload_episode_dataset(
            episode_dataset=combine_episode_datasets(
                expert_episode_dataset=episode_dataset,
                augmentation_episode_dataset=augmentation_result.episode_dataset,
            ),
            name="tacil-dataset",
            run=run,
        )


if __name__ == "__main__":
    main()
