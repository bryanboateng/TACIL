"""Measure how the uncertainty limit maps to augmentation volume.

The limit only enters the backward rollout, never the dynamics model. So one
trained ensemble answers the question for every limit at once, which makes this
about a hundred times cheaper than one policy run per limit.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.data.augmentation.tacil.create_data.anchor_selection import (
    get_anchor_indices,
)
from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    roll_out_backward,
)
from scripts.data.augmentation.tacil.create_data.create_augmented_data import (
    compute_noise_std,
)
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    create_augmented_episodes,
)
from scripts.data.augmentation.tacil.create_data.step_distance import (
    calculate_median_step_distance,
)
from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_ensemble_training import (  # noqa: E501
    train_dynamics_model_ensemble,
)
from scripts.shared.episode_store import download_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seeding import set_seed

logger = create_logger()

# The prediction horizon of the diffusion policy minus one, so that every
# training window starts on a synthetic state.
EXPERT_PART_LENGTH = 15


@dataclass(frozen=True)
class Config:
    seed: int
    wandb_project: str
    expert_dataset_artifact: str
    dynamics: DynamicsConfig

    action_noise_range_fraction: float = 0.03

    limit_step_multiples: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
    )


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project=config.wandb_project,
        job_type="calibration",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        episode_store = download_episode_store(
            artifact_name=config.expert_dataset_artifact, run=run
        )
        dynamics_model_ensemble = train_dynamics_model_ensemble(
            config=config.dynamics,
            episode_store=episode_store,
            seed=config.seed,
            device=device,
            run=run,
        )

        anchor_indices = get_anchor_indices(
            episode_end_indices=episode_store.episode_end_indices,
            expert_part_length=EXPERT_PART_LENGTH,
        )
        anchor_state_batch = torch.as_tensor(episode_store.states[anchor_indices]).to(
            device=device
        )
        anchor_action_batch = torch.as_tensor(episode_store.actions[anchor_indices]).to(
            device=device
        )
        action_noise_std = torch.as_tensor(
            compute_noise_std(
                samples=episode_store.actions,
                noise_range_fraction=config.action_noise_range_fraction,
            )
        ).to(device=device)
        median_step_distance = calculate_median_step_distance(
            episode_store=episode_store, dynamics_model_ensemble=dynamics_model_ensemble
        )
        max_step_count = int(
            np.diff(a=episode_store.episode_end_indices, prepend=0).max()
        )

        logger.info(
            "median expert step distance %.4f, step cap %d, %d anchors",
            median_step_distance,
            max_step_count,
            len(anchor_indices),
        )

        table = wandb.Table(
            columns=[
                "limit_step_multiple",
                "synthetic_fraction",
                "empty_anchor_fraction",
                "chain_length_median",
                "chain_length_p90",
                "chain_length_max",
            ]
        )
        for limit_step_multiple in config.limit_step_multiples:
            # Every limit starts from the same random state, so a longer chain
            # is the limit's doing and not a different noise draw.
            set_seed(seed=config.seed)

            backward_trajectories, _ = roll_out_backward(
                dynamics_model_ensemble=dynamics_model_ensemble,
                anchor_state_batch=anchor_state_batch,
                anchor_action_batch=anchor_action_batch,
                action_noise_std=action_noise_std,
                uncertainty_limit=limit_step_multiple * median_step_distance,
                convergence_tolerance=1e-3 * median_step_distance,
                max_step_count=max_step_count,
            )
            chain_lengths = backward_trajectories.step_count_batch.cpu().numpy()
            if not bool((chain_lengths > 0).any()):
                logger.info("limit %g: no anchor got a step", limit_step_multiple)
                continue

            augmentation_result = create_augmented_episodes(
                backward_trajectories=backward_trajectories,
                episode_store=episode_store,
                anchor_indices=anchor_indices,
                expert_part_length=EXPERT_PART_LENGTH,
            )
            accepted_lengths = chain_lengths[chain_lengths > 0]
            # The same ratio the pipeline logs as
            # 2_data/5_dataset/synthetic_transition_fraction, so the numbers
            # here line up with the ones the sweeps report.
            combined_transition_count = len(episode_store.states) + len(
                augmentation_result.episode_store.states
            )
            row = (
                limit_step_multiple,
                float(accepted_lengths.sum() / combined_transition_count),
                float((chain_lengths == 0).mean()),
                float(np.median(accepted_lengths)),
                float(np.percentile(accepted_lengths, 90)),
                float(accepted_lengths.max()),
            )
            table.add_data(*row)
            logger.info(
                "limit %-6g synthetic %.3f  empty %.3f  length median %.0f p90 %.0f "
                "max %.0f",
                *row,
            )

        run.log(data={"calibration/limit_to_volume": table})


if __name__ == "__main__":
    main()
