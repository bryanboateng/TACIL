import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
import wandb
import yaml
from rich.pretty import pretty_repr
from wandb import Run

from scripts.data.augmentation.ccil import augmentation as ccil
from scripts.data.augmentation.ccil.config import (
    BackwardEulerLabels,
    CcilConfig,
    DynamicsConfig,
    ForwardEulerLabels,
    LabelConfig,
    LearnedSlack,
    NoisyActionLabels,
    SoftSamplingPenalty,
    SpectralNormalization,
)
from scripts.policies.bc import config as bc_config
from scripts.policies.bc.train import train_behavior_cloning_policy
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.state_action_pairs import StateActionPairs

logger = create_logger()


@dataclass(frozen=True)
class Noise:
    size: float = 1e-4

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be positive, but it is {self.size}.")


@dataclass(frozen=True)
class Ccil:
    pass


Augmentation: TypeAlias = Noise | Ccil


@dataclass(frozen=True)
class Config:
    seed: int
    wandb_project: str

    task_config_path: Path
    policy_file_path: Path
    augmentation: Augmentation | None


def train(*, config: Config) -> None:
    logger.info(pretty_repr(config))
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with config.task_config_path.open() as file:
        task_config = yaml.safe_load(file)

    # The config sits in <ccil>/config, and the path it names is relative to
    # <ccil>.
    ccil_directory_path = config.task_config_path.parent.parent
    demonstration_file_path = ccil_directory_path / task_config["data"]["pkl"]
    expert_episodes = _read_ccil_demonstrations(file_path=demonstration_file_path)

    policy_config = _build_policy_config(task_config=task_config, config=config)
    logger.info(pretty_repr(policy_config))

    ccil_config = (
        _build_ccil_config(task_config=task_config)
        if isinstance(config.augmentation, Ccil)
        else None
    )
    if ccil_config is not None:
        logger.info(pretty_repr(ccil_config))

    with wandb.init(
        project=config.wandb_project,
        job_type="ccil-replication",
        config=asdict(obj=config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        run.summary["0_task/1_environment"] = task_config["env"]
        run.summary["0_task/2_demonstration_file"] = demonstration_file_path.name
        run.summary["0_task/3_episode_count"] = len(expert_episodes.episode_end_indices)
        run.summary["0_task/4_transition_count"] = len(expert_episodes.states)
        run.summary["0_task/5_evaluation_noise"] = task_config["eval"]["noise"]

        logger.info("Stage 1/3: building the training data.")
        state_action_pairs = _build_training_data(
            ccil_config=ccil_config,
            task_config=task_config,
            expert_episodes=expert_episodes,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 1/3: done.")

        logger.info("Stage 2/3: training the policy.")
        policy_model = train_behavior_cloning_policy(
            config=policy_config,
            state_action_pairs=state_action_pairs,
            seed=config.seed,
            device=device,
            run=run,
        )
        logger.info("Stage 2/3: done.")

        logger.info("Stage 3/3: exporting the policy.")
        _export_policy(policy_model=policy_model, file_path=config.policy_file_path)
        run.summary["0_task/6_policy_file"] = str(config.policy_file_path)
        logger.info("Stage 3/3: wrote %s.", config.policy_file_path)


# ---- Private --------------------------------------------


def _build_policy_config(*, task_config: dict, config: Config) -> bc_config.Config:
    policy_section = task_config["policy"]
    epoch_count = policy_section["train_epochs"]

    if isinstance(config.augmentation, Ccil) and _balances_data(
        task_config=task_config
    ):
        epoch_count //= task_config["aug"]["num_labels"]

    return bc_config.Config(
        optimization=bc_config.OptimizationConfig(
            batch_size=policy_section["batch_size"],
            epoch_count=epoch_count,
            learning_rate=policy_section["lr"],
        ),
        validation=None,
        observation_noise=(
            config.augmentation.size if isinstance(config.augmentation, Noise) else None
        ),
    )


def _build_ccil_config(*, task_config: dict) -> CcilConfig:
    dynamics_section = task_config["dynamics"]

    if dynamics_section["activation"] != "relu":
        raise ValueError(
            f"Only relu is ported, but the config asks for "
            f"{dynamics_section['activation']}."
        )

    lipschitz_type = dynamics_section["lipschitz_type"]
    match lipschitz_type:
        case "soft_sampling":
            regularization = SoftSamplingPenalty(
                lipschitz_constraint=dynamics_section["lipschitz_constraint"],
                penalty_weight=dynamics_section["soft_lipschitz_penalty_weight"],
                sampling_standard_deviation=dynamics_section[
                    "soft_lipschitz_sampling_eps"
                ],
                sample_count=dynamics_section["soft_lipschitz_n_samples"],
            )
        case "spectral_normalization":
            regularization = SpectralNormalization(
                lipschitz_constraint=dynamics_section["lipschitz_constraint"]
            )
        case "slack":
            regularization = LearnedSlack(
                weight=dynamics_section.get("slack_weight", 1.0)
            )
        case "none":
            regularization = None
        case _:
            raise ValueError(f"lipschitz_type {lipschitz_type} is not ported.")

    label_section = task_config["aug"]
    label_type = label_section["type"]
    match label_type:
        case "noisy_action":
            generation = NoisyActionLabels(
                label_count_per_transition=label_section["num_labels"],
                action_noise_std=label_section["delta"],
            )
        case "backward_euler" | "backward_euler_fast":
            generation = BackwardEulerLabels()
        case "forward_euler":
            generation = ForwardEulerLabels(
                max_iteration_count=label_section["max_iter"]
            )
        case _:
            raise ValueError(f"Label type {label_type} is not ported.")

    epsilon = label_section.get("epsilon")
    return CcilConfig(
        dynamics=DynamicsConfig(
            hidden_layer_sizes=tuple(dynamics_section["layers"]),
            learning_rate=dynamics_section["lr"],
            weight_decay=dynamics_section["weight_decay"],
            batch_size=dynamics_section["batch_size"],
            epoch_count=dynamics_section["train_epochs"],
            regularization=regularization,
        ),
        labels=LabelConfig(
            generation=generation, epsilon=float(epsilon) if epsilon else None
        ),
    )


def _balances_data(*, task_config: dict) -> bool:
    label_section = task_config["aug"]
    return label_section["type"] == "noisy_action" and label_section.get(
        "balance_data", False
    )


def _read_ccil_demonstrations(*, file_path: Path) -> EpisodeStore:
    with file_path.open("rb") as file:
        trajectories = pickle.load(file)

    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    episode_end_indices: list[int] = []
    transition_count = 0

    for index, trajectory in enumerate(iterable=trajectories):
        state_array = np.asarray(trajectory["observations"], dtype=np.float32)
        action_array = np.asarray(trajectory["actions"], dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[:, np.newaxis]

        if state_array.shape[0] != action_array.shape[0]:
            raise ValueError(
                f"Episode {index} holds {state_array.shape[0]} states but "
                f"{action_array.shape[0]} actions."
            )

        state_arrays.append(state_array)
        action_arrays.append(action_array)
        transition_count += state_array.shape[0]
        episode_end_indices.append(transition_count)

    return EpisodeStore(
        states=np.concatenate(state_arrays),
        actions=np.concatenate(action_arrays),
        episode_end_indices=np.array(episode_end_indices, dtype=np.int64),
    )


def _build_training_data(
    *,
    ccil_config: CcilConfig | None,
    task_config: dict,
    expert_episodes: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> StateActionPairs:
    expert_state_action_pairs = StateActionPairs(
        states=expert_episodes.states, actions=expert_episodes.actions
    )
    if ccil_config is None:
        return expert_state_action_pairs

    synthetic_state_action_pairs = ccil.augment(
        config=ccil_config,
        expert_episodes=expert_episodes,
        seed=seed,
        device=device,
        run=run,
    )

    expert_copy_count = (
        task_config["aug"]["num_labels"] + 1
        if _balances_data(task_config=task_config)
        else 1
    )
    return StateActionPairs(
        states=np.concatenate(
            [expert_state_action_pairs.states] * expert_copy_count
            + [synthetic_state_action_pairs.states],
            axis=0,
        ),  # (pair_count, state_dimensionality)
        actions=np.concatenate(
            [expert_state_action_pairs.actions] * expert_copy_count
            + [synthetic_state_action_pairs.actions],
            axis=0,
        ),  # (pair_count, action_dimensionality)
    )


def _export_policy(*, policy_model: torch.nn.Module, file_path: Path) -> None:
    policy_model.eval()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # The model normalizes on its own, so the CCIL authors' code can load
    # this file and call it.
    torch.jit.script(policy_model.to(device=torch.device("cpu"))).save(str(file_path))
