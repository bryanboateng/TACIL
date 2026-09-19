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
from scripts.environments.pusht.state_space import create_state_space
from scripts.shared.distribution_plot import create_distribution_plot
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


_SEED = 0

_WINDOW_COUNT = 1024
_TRAJECTORY_DEPTH = 20
_MEMBER_COUNT = 8
_REPLAY_STEP_COUNT = 100

_STATE_SPACE = create_state_space()
_PLOT_LAYOUT = _STATE_SPACE.plot_layout
_STATE_DIMENSIONALITY = _STATE_SPACE.dimensionality
_ARENA_SIZE = _PLOT_LAYOUT.horizontal_limits[1]
_BLOCK_ORIENTATION_INDEX = _PLOT_LAYOUT.orientation_groups[0].index

_POINT_COUNT = 4_000


def main() -> None:
    rng = np.random.default_rng(seed=_SEED)

    member_trajectories, true_trajectory = _draw_fanning_trajectories(rng=rng)

    previews = {
        "prediction_error_vs_step_count": create_prediction_error_vs_step_count_plot(
            forward_prediction_errors_per_step_count=_draw_growing_errors(
                rng=rng, growth_per_step=1.25
            ),
            backward_prediction_errors_per_step_count=_stop_walks_early(
                prediction_errors=_draw_growing_errors(rng=rng, growth_per_step=1.6),
                rng=rng,
            ),
        ),
        "replay_error": create_replay_error_plot(
            backward_prediction_errors_per_step_count=_stop_walks_early(
                prediction_errors=_draw_growing_errors(rng=rng, growth_per_step=1.6),
                rng=rng,
            ),
            backward_replay_errors_per_step_count=_stop_walks_early(
                prediction_errors=0.2
                * _draw_growing_errors(rng=rng, growth_per_step=1.35),
                rng=rng,
            ),
        ),
        "true_dynamics_vs_model_0": create_true_dynamics_vs_model_plot(
            member_trajectories=member_trajectories,
            true_trajectory=true_trajectory,
            plot_layout=_PLOT_LAYOUT,
            episode_index=0,
        ),
        "nearest_expert_state": _draw_nearest_expert_state_plot(rng=rng),
        "distribution": create_distribution_plot(
            values=rng.lognormal(mean=1.0, sigma=0.8, size=_POINT_COUNT),
            xlabel="made-up distribution",
        ),
    }

    output_directory_path = OUTPUT_DIRECTORY_PATH / "plot_preview"
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
    *, rng: np.random.Generator
) -> tuple[
    npt.NDArray[np.floating],  # (_MEMBER_COUNT, step_count + 1, state_dimensionality)
    npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
]:
    steps = np.arange(_REPLAY_STEP_COUNT + 1)
    progress = steps / _REPLAY_STEP_COUNT

    true_trajectory = np.empty(
        shape=(_REPLAY_STEP_COUNT + 1, _STATE_DIMENSIONALITY), dtype=np.float64
    )
    true_trajectory[:, 0] = 0.25 * _ARENA_SIZE + 0.45 * _ARENA_SIZE * progress
    true_trajectory[:, 1] = 0.30 * _ARENA_SIZE + 0.30 * _ARENA_SIZE * np.sin(
        np.pi * progress
    )
    true_trajectory[:, 2] = 0.35 * _ARENA_SIZE + 0.35 * _ARENA_SIZE * progress
    true_trajectory[:, 3] = 0.45 * _ARENA_SIZE + 0.20 * _ARENA_SIZE * progress**2
    true_trajectory[:, _BLOCK_ORIENTATION_INDEX] = (2.5 * np.pi * progress) % (
        2.0 * np.pi
    )

    drift_per_dimension = np.array([22.0, 22.0, 18.0, 18.0, 0.9])
    drift_scale = progress[None, :, None] ** 1.5
    member_noise = rng.normal(
        loc=0.0, scale=1.0, size=(_MEMBER_COUNT, 1, _STATE_DIMENSIONALITY)
    )

    member_trajectories = (
        true_trajectory[None, :, :] + drift_scale * member_noise * drift_per_dimension
    )
    member_trajectories[..., _BLOCK_ORIENTATION_INDEX] %= 2.0 * np.pi

    return member_trajectories, true_trajectory


if __name__ == "__main__":
    main()
