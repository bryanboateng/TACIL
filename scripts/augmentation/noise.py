import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
import wandb
from rich.pretty import pretty_repr

from scripts.ashared import compute_config_hash, create_logger
from scripts.shared.data import (
    DemonstrationData,
    load_demonstration_data_from_artifact,
    load_zarr,
    write_zarr,
)

logger = create_logger()


@dataclass
class Config:
    # Fraction of episodes to augment with noise (0, 1]
    fraction: float
    use_artifact: str
    # Noise standard deviation as a fraction of each state dimension's range.
    # E.g. 0.02 = 2% of the observed min-max range per dimension.
    noise_scale: float = 0.02
    # If True, noise std is computed per state dimension from the data.
    # If False, a single std is computed from the global state range.
    per_dimension_noise: bool = True
    seed: int = 42


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config):
    rng = np.random.default_rng(seed=config.seed)

    with wandb.init(
        project="abc", job_type="augmentation-noisy", config=asdict(config)
    ) as run:
        artifact = run.use_artifact(artifact_or_name=config.use_artifact)

        demonstration_data = load_demonstration_data_from_artifact(artifact=artifact, logger=logger)

        logger.info(
            "Applying noise augmentation to %g%% of episodes (noise_scale=%.4f, per_dim=%s) ...",
            config.fraction * 100,
            config.noise_scale,
            config.per_dimension_noise,
        )

        augmented_data = _augment_with_noise(
            demonstration_data=demonstration_data,
            fraction=config.fraction,
            noise_scale=config.noise_scale,
            rng=rng,
        )

        logger.info("Noise augmentation completed.")
        logger.info("# Transitions (unchanged): %d", len(augmented_data.states))
        logger.info(
            "# Episodes (unchanged): %d", len(augmented_data.episode_end_indices)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_zarr_path = Path(tmpdir) / "dataset.zarr"

            write_zarr(
                zarr_file_path=output_zarr_path, demonstration_data=augmented_data
            )

            new_artifact = wandb.Artifact(
                name=f"noisy-dataset-{compute_config_hash(config=asdict(config))}",
                type="dataset",
            )

            new_artifact.add_dir(str(output_zarr_path))

            run.log_artifact(new_artifact)


def _augment_with_noise(
    demonstration_data: DemonstrationData,
    fraction: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> DemonstrationData:
    """
    Augment a fraction of episodes by adding Gaussian noise to states.

    The noise std is computed as `noise_scale * range` where range is either
    per-dimension (if per_dimension_noise=True) or global across all dimensions.

    Actions are left untouched to preserve demonstration quality.
    The episode structure (episode_end_indices) is unchanged since we augment
    in-place rather than duplicating episodes.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    episode_end_indices = demonstration_data.episode_end_indices
    episode_count = len(episode_end_indices)

    # --- Compute noise std from the full dataset (before any augmentation) ---
    states = demonstration_data.states  # shape: (N, state_dim)

    # std per state dimension: noise_scale * (max - min) per dim
    state_min = states.min(axis=0)  # (state_dim,)
    state_max = states.max(axis=0)  # (state_dim,)
    noise_std = noise_scale * (state_max - state_min)  # (state_dim,)
    logger.info("Per-dimension noise std: %s", noise_std)

    # --- Select which episodes get augmented ---
    selected_episode_count = max(1, int(round(episode_count * fraction)))

    selected_episode_indices = rng.choice(
        a=episode_count, size=selected_episode_count, replace=False
    )

    logger.info(
        "Augmenting %d / %d episodes with noise.", selected_episode_count, episode_count
    )

    # --- Compute episode start indices ---
    episode_start_indices = np.concatenate(
        (np.zeros(shape=1, dtype=episode_end_indices.dtype), episode_end_indices[:-1])
    )

    # --- Apply noise to selected episodes ---
    augmented_states = states.copy()

    for episode_index in selected_episode_indices:
        start: int = episode_start_indices[episode_index]
        end: int = episode_end_indices[episode_index]
        episode_length = end - start

        noise = rng.normal(
            loc=0.0, scale=noise_std, size=(episode_length, states.shape[1])
        ).astype(np.float32)

        augmented_states[start:end] += noise

    return DemonstrationData(
        states=augmented_states,
        actions=demonstration_data.actions.copy(),
        episode_end_indices=episode_end_indices.copy(),
    )


if __name__ == "__main__":
    main()
