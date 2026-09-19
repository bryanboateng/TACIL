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
    create_true_dynamics_vs_model_plot,
)
from scripts.environments.plot_layout import PlotLayout
from scripts.environments.registry import EnvironmentName, create_environment
from scripts.environments.state_space import StateSpace
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

_FULL_TURN = 2.0 * np.pi

_DRIFT_FRACTION = 0.06


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
        member_trajectories, true_trajectory = _draw_fanning_trajectories(
            rng=np.random.default_rng(seed=_SEED),
            state_space=environment.state_space,
            plot_layout=environment.plot_layout,
        )
        _save_previews(
            previews={
                "true_dynamics_vs_model_0": create_true_dynamics_vs_model_plot(
                    member_trajectories=member_trajectories,
                    true_trajectory=true_trajectory,
                    plot_layout=environment.plot_layout,
                    episode_index=0,
                )
            },
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


def _draw_fanning_trajectories(
    *, rng: np.random.Generator, state_space: StateSpace, plot_layout: PlotLayout
) -> tuple[
    npt.NDArray[np.floating],  # (_MEMBER_COUNT, step_count + 1, state_dimensionality)
    npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
]:
    progress = np.arange(_REPLAY_STEP_COUNT + 1) / _REPLAY_STEP_COUNT

    true_trajectory = np.zeros(
        shape=(_REPLAY_STEP_COUNT + 1, state_space.dimensionality), dtype=np.float64
    )
    drift_per_entry = np.zeros(state_space.dimensionality)

    horizontal_start, horizontal_end = plot_layout.horizontal_limits
    vertical_start, vertical_end = plot_layout.vertical_limits
    horizontal_span = horizontal_end - horizontal_start
    vertical_span = vertical_end - vertical_start

    for group_index, position_group in enumerate(iterable=plot_layout.position_groups):
        phase = 0.12 * group_index
        true_trajectory[:, position_group.entries] = np.stack(
            arrays=[
                horizontal_start + horizontal_span * (0.12 + 0.76 * progress),
                vertical_start
                + vertical_span
                * (0.12 + 0.76 * np.sin(np.pi * (progress + phase)) ** 2),
            ],
            axis=-1,
        )  # (step_count + 1, 2)
        drift_per_entry[position_group.entries] = _DRIFT_FRACTION * np.array(
            object=[horizontal_span, vertical_span]
        )

    for group_index, orientation_group in enumerate(
        iterable=plot_layout.orientation_groups
    ):
        true_trajectory[:, orientation_group.index] = (
            (1.25 + 0.75 * group_index) * _FULL_TURN * progress
        ) % _FULL_TURN
        drift_per_entry[orientation_group.index] = _DRIFT_FRACTION * _FULL_TURN

    member_trajectories = true_trajectory[None] + (
        progress[None, :, None] ** 1.5
        * rng.normal(
            loc=0.0, scale=1.0, size=(_MEMBER_COUNT, 1, state_space.dimensionality)
        )
        * drift_per_entry
    )  # (_MEMBER_COUNT, step_count + 1, state_dimensionality)
    for orientation_group in plot_layout.orientation_groups:
        member_trajectories[..., orientation_group.index] %= _FULL_TURN

    return member_trajectories, true_trajectory


if __name__ == "__main__":
    main()
