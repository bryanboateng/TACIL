"""Render every panel from made-up numbers, to look at the design without
training a model first.

The numbers are shaped like the real ones — errors that grow with depth,
backward chains that stop early, an ensemble that fans out — but nothing here
measures anything.
"""

import numpy as np
from numpy import typing as npt

from scripts.data.augmentation.tacil.create_dynamics.statistics import panels
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.pusht_state import ARENA_SIZE, BLOCK_ORIENTATION_INDEX

logger = create_logger()

_OUTPUT_DIRECTORY_PATH = OUTPUT_DIRECTORY_PATH / "panel_preview"

_SEED = 0

_CHAIN_COUNT = 1024
_CHAIN_DEPTH = 20
_MEMBER_COUNT = 8
_REPLAY_STEP_COUNT = 100

_STATE_DIMENSIONALITY = 5


def main() -> None:
    rng = np.random.default_rng(seed=_SEED)

    forward_member_trajectories, _, true_trajectory = _fanning_replay(rng=rng)

    images = {
        "error_over_depth": panels.error_over_depth_panel(
            forward_errors_per_depth=_growing_errors(rng=rng, growth_per_step=1.25),
            backward_errors_per_depth=_stop_chains_early(
                true_errors=_growing_errors(rng=rng, growth_per_step=1.6), rng=rng
            ),
        ),
        "replay_episode_0": panels.replay_panel(
            forward_member_trajectories=forward_member_trajectories,
            true_trajectory=true_trajectory,
            episode_index=0,
        ),
    }

    _OUTPUT_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image_path = _OUTPUT_DIRECTORY_PATH / f"{name}.png"
        image.image.save(image_path)
        logger.info("Wrote %s", image_path)


def _growing_errors(
    *, rng: np.random.Generator, growth_per_step: float
) -> npt.NDArray[np.floating]:  # (_CHAIN_COUNT, _CHAIN_DEPTH)
    """Error that compounds with depth, spread over chains by a lognormal
    factor — the log-scale axes of the panels need a spread, not a band of
    equal width."""
    depths = np.arange(1, _CHAIN_DEPTH + 1)  # (_CHAIN_DEPTH,)
    median_per_depth = 0.01 * growth_per_step ** (depths - 1)  # (_CHAIN_DEPTH,)
    per_chain_factors = rng.lognormal(
        mean=0.0, sigma=0.55, size=(_CHAIN_COUNT, _CHAIN_DEPTH)
    )
    return median_per_depth * per_chain_factors


def _stop_chains_early(
    *, true_errors: npt.NDArray[np.floating], rng: np.random.Generator
) -> npt.NDArray[np.floating]:  # (_CHAIN_COUNT, _CHAIN_DEPTH)
    """A backward chain ends at its first non-converged solve and is NaN from
    there on, so the deeper columns hold fewer and fewer chains."""
    survived_depths = rng.geometric(p=0.12, size=_CHAIN_COUNT)  # (_CHAIN_COUNT,)
    depths = np.arange(1, _CHAIN_DEPTH + 1)  # (_CHAIN_DEPTH,)

    stopped_mask = depths > survived_depths[:, None]
    return np.where(stopped_mask, np.nan, true_errors)


def _fanning_replay(
    *, rng: np.random.Generator
) -> tuple[
    npt.NDArray[np.floating],  # (_MEMBER_COUNT, step_count + 1, state_dimensionality)
    npt.NDArray[np.floating],  # (_MEMBER_COUNT, step_count + 1, state_dimensionality)
    npt.NDArray[np.floating],  # (step_count + 1, state_dimensionality)
]:
    """A recorded arc, and members that drift off it further the longer they
    predict on their own — forward from the first state, backward from the
    last one."""
    steps = np.arange(_REPLAY_STEP_COUNT + 1)
    progress = steps / _REPLAY_STEP_COUNT

    true_trajectory = np.empty(
        shape=(_REPLAY_STEP_COUNT + 1, _STATE_DIMENSIONALITY), dtype=np.float64
    )
    true_trajectory[:, 0] = 0.25 * ARENA_SIZE + 0.45 * ARENA_SIZE * progress
    true_trajectory[:, 1] = 0.30 * ARENA_SIZE + 0.30 * ARENA_SIZE * np.sin(
        np.pi * progress
    )
    true_trajectory[:, 2] = 0.35 * ARENA_SIZE + 0.35 * ARENA_SIZE * progress
    true_trajectory[:, 3] = 0.45 * ARENA_SIZE + 0.20 * ARENA_SIZE * progress**2
    # Wrapped to [0, 2*pi), like the real state: the block spins through
    # several full turns over the episode.
    true_trajectory[:, BLOCK_ORIENTATION_INDEX] = (2.5 * np.pi * progress) % (
        2.0 * np.pi
    )

    drift_per_dimension = np.array([22.0, 22.0, 18.0, 18.0, 0.9])
    member_trajectories = []
    for drift_scale in (
        # Forward: at step 0 every member sits on the recorded state, and
        # nothing has compounded yet. Backward mirrors that around the end.
        progress[None, :, None] ** 1.5,
        (1.0 - progress)[None, :, None] ** 1.5,
    ):
        member_noise = rng.normal(
            loc=0.0, scale=1.0, size=(_MEMBER_COUNT, 1, _STATE_DIMENSIONALITY)
        )
        trajectories = (
            true_trajectory[None, :, :]
            + drift_scale * member_noise * drift_per_dimension
        )
        trajectories[..., BLOCK_ORIENTATION_INDEX] %= 2.0 * np.pi
        member_trajectories.append(trajectories)

    forward_member_trajectories, backward_member_trajectories = member_trajectories
    # Half the members give up before reaching the start, the way a solve that
    # stops converging leaves the rest of the walk empty.
    backward_member_trajectories[: _MEMBER_COUNT // 2, : _REPLAY_STEP_COUNT // 4, :] = (
        np.nan
    )

    return forward_member_trajectories, backward_member_trajectories, true_trajectory


if __name__ == "__main__":
    main()
