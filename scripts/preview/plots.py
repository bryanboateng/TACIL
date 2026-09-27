from pathlib import Path

import numpy as np
import wandb
from numpy import typing as npt

from scripts.data.augmentation.tacil.create_data.nearest_expert_state_plot import (
    create_nearest_expert_state_plot,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics.plots import (
    create_prediction_error_vs_step_count_plot,
    create_replay_error_plot,
)
from scripts.environments.peg_insertion import plot_preview as peg_insertion_preview
from scripts.environments.pusht import plot_preview as pusht_preview
from scripts.environments.registry import EnvironmentName, create_environment
from scripts.shared.distribution_plot import create_distribution_plot
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


_SEED = 0

_WINDOW_COUNT = 1024
_TRAJECTORY_DEPTH = 20
_MEMBER_COUNT = 8
_REPLAY_STEP_COUNT = 100

_POINT_COUNT = 4_000


def main() -> None:
    rng = np.random.default_rng(seed=_SEED)
    output_directory_path = OUTPUT_DIRECTORY_PATH / "plot_preview"

    _save_previews(
        previews={
            "prediction_error_vs_step_count": (
                create_prediction_error_vs_step_count_plot(
                    forward_prediction_errors_per_step_count=_draw_growing_errors(
                        rng=rng, growth_per_step=1.25
                    ),
                    backward_prediction_errors_per_step_count=_stop_walks_early(
                        prediction_errors=_draw_growing_errors(
                            rng=rng, growth_per_step=1.6
                        ),
                        rng=rng,
                    ),
                )
            ),
            "replay_error": create_replay_error_plot(
                backward_prediction_errors_per_step_count=_stop_walks_early(
                    prediction_errors=_draw_growing_errors(
                        rng=rng, growth_per_step=1.6
                    ),
                    rng=rng,
                ),
                backward_replay_errors_per_step_count=_stop_walks_early(
                    prediction_errors=0.2
                    * _draw_growing_errors(rng=rng, growth_per_step=1.35),
                    rng=rng,
                ),
            ),
            "nearest_expert_state": _draw_nearest_expert_state_plot(rng=rng),
            "distribution": create_distribution_plot(
                values=rng.lognormal(mean=1.0, sigma=0.8, size=_POINT_COUNT),
                xlabel="made-up distribution",
            ),
        },
        output_directory_path=output_directory_path,
    )

    for environment_name in EnvironmentName:
        environment = create_environment(name=environment_name)
        match environment_name:
            case EnvironmentName.PUSH_T:
                draw_trajectories = pusht_preview.draw_trajectories
            case EnvironmentName.PEG_INSERTION:
                draw_trajectories = peg_insertion_preview.draw_trajectories

        member_trajectories, true_trajectory = draw_trajectories(
            rng=np.random.default_rng(seed=_SEED),
            step_count=_REPLAY_STEP_COUNT + 1,
            member_count=_MEMBER_COUNT,
        )
        _save_previews(
            previews=environment.tacil_dynamics_plotter(
                member_trajectories=member_trajectories,
                true_trajectory=true_trajectory,
                episode_index=0,
            ),
            output_directory_path=output_directory_path / environment_name.value,
        )


def _save_previews(
    *, previews: dict[str, wandb.Image], output_directory_path: Path
) -> None:
    output_directory_path.mkdir(parents=True, exist_ok=True)
    for name, preview in previews.items():
        image = preview.image
        if image is None:
            raise RuntimeError(f"The {name} preview unexpectedly holds no image.")
        image_path = output_directory_path / f"{name}.png"
        image.save(image_path)
        logger.info("Wrote %s", image_path)


def _draw_nearest_expert_state_plot(*, rng: np.random.Generator) -> wandb.Image:
    expert_state_distances = rng.lognormal(mean=0.0, sigma=0.4, size=_POINT_COUNT)
    synthetic_state_distances = rng.lognormal(mean=0.7, sigma=0.7, size=_POINT_COUNT)
    return create_nearest_expert_state_plot(
        synthetic_state_distances=synthetic_state_distances,
        synthetic_action_distances=(
            6.0
            * synthetic_state_distances
            * rng.lognormal(mean=0.0, sigma=0.5, size=_POINT_COUNT)
        ),
        expert_state_distances=expert_state_distances,
        expert_action_distances=(
            6.0
            * expert_state_distances
            * rng.lognormal(mean=0.0, sigma=0.5, size=_POINT_COUNT)
        ),
    )


def _draw_growing_errors(
    *, rng: np.random.Generator, growth_per_step: float
) -> npt.NDArray[np.floating]:  # (_WINDOW_COUNT, _TRAJECTORY_DEPTH)
    depths = np.arange(1, _TRAJECTORY_DEPTH + 1)  # (_TRAJECTORY_DEPTH,)
    median_per_depth = 0.01 * growth_per_step ** (depths - 1)  # (_TRAJECTORY_DEPTH,)
    per_window_factors = rng.lognormal(
        mean=0.0, sigma=0.55, size=(_WINDOW_COUNT, _TRAJECTORY_DEPTH)
    )
    return median_per_depth * per_window_factors


def _stop_walks_early(
    *, prediction_errors: npt.NDArray[np.floating], rng: np.random.Generator
) -> npt.NDArray[np.floating]:  # (_WINDOW_COUNT, _TRAJECTORY_DEPTH)
    survived_depths = rng.geometric(p=0.12, size=_WINDOW_COUNT)  # (_WINDOW_COUNT,)
    depths = np.arange(1, _TRAJECTORY_DEPTH + 1)  # (_TRAJECTORY_DEPTH,)

    stopped_mask = depths > survived_depths[:, None]
    return np.where(stopped_mask, np.nan, prediction_errors)


if __name__ == "__main__":
    main()
