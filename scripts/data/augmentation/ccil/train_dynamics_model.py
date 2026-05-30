import matplotlib

matplotlib.use(backend="Agg")

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
import tyro
import wandb
import zarr
from matplotlib import pyplot as plt
from numpy import typing as npt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from torch import vmap
from torch.func import jacrev
from tqdm import tqdm
from wandb import Run

from scripts.ashared import create_logger, load_push_t_zarr, log_device_info, set_seed
from scripts.shared.paths import CCIL_ROOT

logger = create_logger()


Episode = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2_048
    epoch_count: int = 500


@dataclass
class Config:
    lipschitz_constant_constraint: float = 1.0
    lipschitz_penalty_weight: float = 1.0
    estimation_method: Literal["jacobian", "sampling"] = "sampling"
    noise_scale: float = 0.05
    perturbation_count: int = 10


@dataclass(frozen=True)
class LipschitzConfig:
    config: Config
    include_slack: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    expert_holdout_fraction: float = 0.1
    split_seed: int = 0
    rollout_horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    random_rollouts_data_path: Path = field(
        default_factory=lambda: CCIL_ROOT / "random_rollouts.zarr"
    )


class DynamicsModel(torch.nn.Module):
    _state_mean: torch.Tensor
    _state_std: torch.Tensor
    _action_mean: torch.Tensor
    _action_std: torch.Tensor
    _next_state_residual_mean: torch.Tensor
    _next_state_residual_std: torch.Tensor

    def __init__(
        self,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_dimensionalities: tuple[int, ...],
    ):
        super().__init__()

        self.hidden_dimensionalities = hidden_dimensionalities
        self.state_dimensionality = state_dimensionality
        self.action_dimensionality = action_dimensionality

        self._net = _build_mlp(
            input_dimensionality=state_dimensionality + action_dimensionality,
            output_dimensionality=state_dimensionality,
            hidden_dimensionalities=self.hidden_dimensionalities,
        )

        self.register_buffer(
            name="_state_mean", tensor=torch.zeros(state_dimensionality)
        )
        self.register_buffer(name="_state_std", tensor=torch.ones(state_dimensionality))
        self.register_buffer(
            name="_action_mean", tensor=torch.zeros(action_dimensionality)
        )
        self.register_buffer(
            name="_action_std", tensor=torch.ones(action_dimensionality)
        )
        self.register_buffer(
            name="_next_state_residual_mean", tensor=torch.zeros(state_dimensionality)
        )
        self.register_buffer(
            name="_next_state_residual_std", tensor=torch.ones(state_dimensionality)
        )

    @torch.no_grad()
    def compute_normalization_stats(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
    ) -> None:
        eps: float = 1e-6
        self._state_mean.copy_(state_batch.mean(dim=0))
        self._state_std.copy_(state_batch.std(dim=0).clamp_min(min=eps))

        self._action_mean.copy_(action_batch.mean(dim=0))
        self._action_std.copy_(action_batch.std(dim=0).clamp_min(min=eps))

        residual_batch = next_state_batch - state_batch
        self._next_state_residual_mean.copy_(residual_batch.mean(dim=0))
        self._next_state_residual_std.copy_(
            residual_batch.std(dim=0).clamp_min(min=eps)
        )

    @property
    def state_std(self) -> torch.Tensor:
        return self._state_std

    @property
    def action_std(self) -> torch.Tensor:
        return self._action_std

    def _predict_next_state_residual(
        self, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> torch.Tensor:
        normalized_state_batch = self._normalize(
            input_batch=state_batch,
            mean_batch=self._state_mean,
            std_batch=self._state_std,
        )

        normalized_action_batch = self._normalize(
            input_batch=action_batch,
            mean_batch=self._action_mean,
            std_batch=self._action_std,
        )

        normalized_residual_batch = self._net(
            torch.cat(tensors=[normalized_state_batch, normalized_action_batch], dim=-1)
        )
        return self._denormalize(
            input_batch=normalized_residual_batch,
            mean_batch=self._next_state_residual_mean,
            std_batch=self._next_state_residual_std,
        )

    def forward(
        self, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> torch.Tensor:
        return state_batch + self._predict_next_state_residual(
            state_batch=state_batch, action_batch=action_batch
        )

    @staticmethod
    def _normalize(
        input_batch: torch.Tensor, mean_batch: torch.Tensor, std_batch: torch.Tensor
    ) -> torch.Tensor:
        return (input_batch - mean_batch) / std_batch

    @staticmethod
    def _denormalize(
        input_batch: torch.Tensor, mean_batch: torch.Tensor, std_batch: torch.Tensor
    ) -> torch.Tensor:
        return input_batch * std_batch + mean_batch


@dataclass
class DynamicsLoss:
    total: torch.Tensor
    prediction: torch.Tensor
    lipschitz_regularization: torch.Tensor


class LossFunction(Protocol):
    def __call__(
        self,
        dynamics_model: DynamicsModel,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
    ) -> DynamicsLoss: ...


def main(
    training: TrainingConfig,
    lipschitz: LipschitzConfig,
    evaluation: EvaluationConfig,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (256, 256),
) -> None:
    set_seed(seed=seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project="asdf",
        job_type="dynamics_model",
        config={
            "training": asdict(training),
            "lipschitz": asdict(lipschitz),
            "evaluation": {
                **asdict(evaluation),
                "random_rollouts_data_path": str(evaluation.random_rollouts_data_path),
            },
            "seed": seed,
            "hidden_layer_sizes": list(hidden_layer_sizes),
        },
    ) as run:
        assert run.name is not None

        output_directory_path = CCIL_ROOT / "dynamics" / run.name
        output_directory_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_directory_path)

        train_episodes, expert_holdout_episodes = train_test_split(
            load_episodes(),
            test_size=evaluation.expert_holdout_fraction,
            random_state=evaluation.split_seed,
            shuffle=True,
        )
        logger.info(
            "Train episodes: %d  Expert-holdout episodes: %d",
            len(train_episodes),
            len(expert_holdout_episodes),
        )

        train_states, train_actions, train_next_states = concatenate_episodes(
            episodes=train_episodes
        )
        (expert_holdout_states, expert_holdout_actions, expert_holdout_next_states) = (
            concatenate_episodes(episodes=expert_holdout_episodes)
        )

        dynamics_model, importance_head = build_models(
            hidden_layer_sizes=hidden_layer_sizes,
            include_slack=lipschitz.include_slack,
            state_dimensionality=train_states.shape[1],
            action_dimensionality=train_actions.shape[1],
            device=device,
        )

        loss_function = create_loss_function(
            config=lipschitz.config, importance_network=importance_head
        )

        fit_dynamics(
            dynamics_model=dynamics_model,
            importance_head=importance_head,
            states=train_states,
            actions=train_actions,
            next_states=train_next_states,
            batch_size=training.batch_size,
            epoch_count=training.epoch_count,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            loss_function=loss_function,
            run=run,
        )

        save_checkpoint(
            dynamics_model=dynamics_model,
            importance_head=importance_head,
            output_directory_path=output_directory_path,
            run=run,
        )

        random_rollout_episodes = load_random_rollout_episodes(
            path=evaluation.random_rollouts_data_path
        )

        expert_holdout_rollout_errors = eval_multi_step_rollout_error(
            dynamics_model=dynamics_model,
            episodes=expert_holdout_episodes,
            horizon_lengths=evaluation.rollout_horizons,
        )

        random_rollout_errors = eval_multi_step_rollout_error(
            dynamics_model=dynamics_model,
            episodes=random_rollout_episodes,
            horizon_lengths=evaluation.rollout_horizons,
        )

        save_evaluation_summary(
            expert_holdout_rollout_errors=expert_holdout_rollout_errors,
            random_rollout_errors=random_rollout_errors,
            run=run,
        )

        save_multi_step_rollout_plot(
            expert_holdout_rollout_errors=expert_holdout_rollout_errors,
            random_rollout_errors=random_rollout_errors,
            output_directory_path=output_directory_path,
            run=run,
        )

        save_lipschitz_distribution_plot(
            dynamics_model=dynamics_model,
            lipschitz_constant_constraint=lipschitz.config.lipschitz_constant_constraint,
            expert_holdout_states=expert_holdout_states,
            expert_holdout_actions=expert_holdout_actions,
            random_rollout_episodes=random_rollout_episodes,
            output_directory_path=output_directory_path,
            run=run,
        )


def build_models(
    hidden_layer_sizes: tuple[int, ...],
    include_slack: bool,
    state_dimensionality: int,
    action_dimensionality: int,
    device: torch.device,
) -> tuple[DynamicsModel, torch.nn.Module | None]:
    dynamics_model = DynamicsModel(
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        hidden_dimensionalities=hidden_layer_sizes,
    ).to(device=device)

    importance_head = None
    if include_slack:
        importance_head = make_importance_head(
            state_dimensionality=state_dimensionality,
            action_dimensionality=action_dimensionality,
            hidden_size=hidden_layer_sizes,
        ).to(device=device)

    return dynamics_model, importance_head


def _build_mlp(
    input_dimensionality: int,
    output_dimensionality: int,
    hidden_dimensionalities: tuple[int, ...],
) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []

    current_layer_input_dimensionality = input_dimensionality
    for hidden_dimensionality in hidden_dimensionalities:
        layers.append(
            torch.nn.Linear(
                in_features=current_layer_input_dimensionality,
                out_features=hidden_dimensionality,
            )
        )
        layers.append(torch.nn.ReLU())
        current_layer_input_dimensionality = hidden_dimensionality

    layers.append(
        torch.nn.Linear(
            in_features=current_layer_input_dimensionality,
            out_features=output_dimensionality,
        )
    )
    return torch.nn.Sequential(*layers)


def make_importance_head(
    state_dimensionality: int, action_dimensionality: int, hidden_size: tuple[int, ...]
) -> torch.nn.Sequential:
    return _build_mlp(
        input_dimensionality=state_dimensionality + action_dimensionality,
        output_dimensionality=1,
        hidden_dimensionalities=hidden_size,
    )


def create_loss_function(
    config: Config, importance_network: torch.nn.Module | None
) -> LossFunction:

    if importance_network is not None:

        def loss_function(
            dynamics_model: DynamicsModel,
            state_batch: torch.Tensor,
            action_batch: torch.Tensor,
            next_state_batch: torch.Tensor,
        ) -> DynamicsLoss:
            assert importance_network is not None
            return compute_importance_weighted_lipschitz_regularized_dynamics_loss(
                dynamics_model=dynamics_model,
                state_batch=state_batch,
                action_batch=action_batch,
                next_state_batch=next_state_batch,
                importance_network=importance_network,
                config=config,
            )

        return loss_function

    else:

        def loss_function(
            dynamics_model: DynamicsModel,
            state_batch: torch.Tensor,
            action_batch: torch.Tensor,
            next_state_batch: torch.Tensor,
        ) -> DynamicsLoss:
            return compute_lipschitz_regularized_dynamics_loss(
                dynamics_model=dynamics_model,
                state_batch=state_batch,
                action_batch=action_batch,
                next_state_batch=next_state_batch,
                config=config,
            )

        return loss_function


def load_episodes() -> list[Episode]:
    flat_states, flat_actions, episode_ends = load_push_t_zarr()
    return _split_flat_arrays_into_episodes(
        flat_states=flat_states, flat_actions=flat_actions, episode_ends=episode_ends
    )


def load_random_rollout_episodes(path: Path) -> list[Episode]:
    store = zarr.open(store=path, mode="r")
    flat_states = np.asarray(store["data/state"][:], dtype=np.float32)
    flat_actions = np.asarray(store["data/action"][:], dtype=np.float32)
    episode_ends = np.asarray(store["meta/episode_ends"][:], dtype=np.int64)
    return _split_flat_arrays_into_episodes(
        flat_states=flat_states, flat_actions=flat_actions, episode_ends=episode_ends
    )


def _split_flat_arrays_into_episodes(
    flat_states: npt.NDArray[np.float32],
    flat_actions: npt.NDArray[np.float32],
    episode_ends: npt.NDArray[np.int64],
) -> list[Episode]:
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episodes: list[Episode] = []
    for episode_start, episode_end in zip(episode_starts, episode_ends):
        episode_states = flat_states[episode_start:episode_end]
        episode_actions = flat_actions[episode_start:episode_end]
        episodes.append((episode_states[:-1], episode_actions[:-1], episode_states[1:]))
    return episodes


def compute_lipschitz_regularized_dynamics_loss(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    next_state_batch: torch.Tensor,
    config: Config,
) -> DynamicsLoss:
    predicted_next_state_batch = dynamics_model(state_batch, action_batch)
    prediction_loss = torch.nn.functional.mse_loss(
        input=predicted_next_state_batch, target=next_state_batch
    )

    lipschitz_violations = _estimate_local_lipschitz_violations(
        dynamics_model=dynamics_model,
        state_batch=state_batch,
        action_batch=action_batch,
        config=config,
    )

    lipschitz_regularization_loss = (
        lipschitz_violations.mean() * config.lipschitz_penalty_weight
    )

    return DynamicsLoss(
        total=prediction_loss + lipschitz_regularization_loss,
        prediction=prediction_loss.detach(),
        lipschitz_regularization=lipschitz_regularization_loss.detach(),
    )


def compute_importance_weighted_lipschitz_regularized_dynamics_loss(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    next_state_batch: torch.Tensor,
    importance_network: torch.nn.Module,
    config: Config,
) -> DynamicsLoss:
    predicted_next_state_batch = dynamics_model(state_batch, action_batch)
    prediction_loss = torch.nn.functional.mse_loss(
        input=predicted_next_state_batch, target=next_state_batch
    )

    lipschitz_violations = _estimate_local_lipschitz_violations(
        dynamics_model=dynamics_model,
        state_batch=state_batch,
        action_batch=action_batch,
        config=config,
    )

    per_sample_importance = torch.nn.functional.softplus(
        importance_network(torch.cat([state_batch, action_batch], dim=-1)).squeeze(-1)
    )  # [batch]

    importance_regularization = 1.0 - torch.exp(
        input=-0.1 * per_sample_importance.abs().mean()
    )
    weighted_lipschitz_violations = lipschitz_violations * per_sample_importance

    lipschitz_regularization_loss = config.lipschitz_penalty_weight * (
        weighted_lipschitz_violations.mean() + importance_regularization
    )

    return DynamicsLoss(
        total=prediction_loss + lipschitz_regularization_loss,
        prediction=prediction_loss.detach(),
        lipschitz_regularization=lipschitz_regularization_loss.detach(),
    )


def _estimate_local_lipschitz_violations(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    if config.estimation_method == "jacobian":
        return _estimate_local_lipschitz_violations_jacobian(
            dynamics_model=dynamics_model,
            state_batch=state_batch,
            action_batch=action_batch,
            lipschitz_constant_constraint=config.lipschitz_constant_constraint,
        )
    if config.estimation_method == "sampling":
        return _estimate_local_lipschitz_violations_sampling(
            dynamics_model=dynamics_model,
            state_batch=state_batch,
            action_batch=action_batch,
            lipschitz_constant_constraint=config.lipschitz_constant_constraint,
            noise_scale=config.noise_scale,
            perturbation_count=config.perturbation_count,
        )
    raise ValueError(f"Unknown estimation method: {config.estimation_method!r}")


def _estimate_local_lipschitz_violations_jacobian(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    lipschitz_constant_constraint: float,
) -> torch.Tensor:
    state_dimensionality = state_batch.shape[-1]
    model_input_batch = torch.cat([state_batch, action_batch], dim=-1)

    def predict_from_concatenated_input(model_input: torch.Tensor) -> torch.Tensor:
        return dynamics_model(
            model_input[..., :state_dimensionality],
            model_input[..., state_dimensionality:],
        )

    jacobians = vmap(jacrev(predict_from_concatenated_input))(model_input_batch)
    spectral_norms = torch.linalg.matrix_norm(jacobians, ord=2)

    return torch.relu(spectral_norms - lipschitz_constant_constraint)


def _estimate_local_lipschitz_violations_sampling(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    lipschitz_constant_constraint: float,
    noise_scale: float,
    perturbation_count: int,
) -> torch.Tensor:
    batch_size = state_batch.shape[0]
    state_dimensionality = state_batch.shape[-1]
    action_dimensionality = action_batch.shape[-1]
    device = state_batch.device

    predicted_next_state_batch = dynamics_model(state_batch, action_batch)  # (B, D_s)

    state_noise_batch = (
        torch.randn(
            (perturbation_count, batch_size, state_dimensionality), device=device
        )
        * dynamics_model.state_std
        * noise_scale
    )
    action_noise_batch = (
        torch.randn(
            (perturbation_count, batch_size, action_dimensionality), device=device
        )
        * dynamics_model.action_std
        * noise_scale
    )

    perturbed_state_batch = state_batch.unsqueeze(0) + state_noise_batch
    perturbed_action_batch = action_batch.unsqueeze(0) + action_noise_batch

    perturbed_predictions = dynamics_model(
        perturbed_state_batch.reshape(-1, state_dimensionality),
        perturbed_action_batch.reshape(-1, action_dimensionality),
    ).reshape(perturbation_count, batch_size, state_dimensionality)

    output_perturbation_norms = torch.linalg.norm(
        perturbed_predictions - predicted_next_state_batch.unsqueeze(0), dim=-1
    )
    input_perturbation_norms = torch.linalg.norm(
        torch.cat([state_noise_batch, action_noise_batch], dim=-1), dim=-1
    )

    violations_per_perturbation = torch.relu(
        output_perturbation_norms
        - lipschitz_constant_constraint * input_perturbation_norms
    )
    return violations_per_perturbation.mean(dim=0)  # (B,)


def fit_dynamics(
    dynamics_model: DynamicsModel,
    importance_head: torch.nn.Module | None,
    states: npt.NDArray[np.float32],
    actions: npt.NDArray[np.float32],
    next_states: npt.NDArray[np.float32],
    batch_size: int,
    epoch_count: int,
    learning_rate: float,
    weight_decay: float,
    loss_function: LossFunction,
    run: Run,
) -> None:
    model_device = next(dynamics_model.parameters()).device
    state_batch = torch.from_numpy(states).to(device=model_device)
    action_batch = torch.from_numpy(actions).to(device=model_device)
    next_state_batch = torch.from_numpy(next_states).to(device=model_device)
    assert state_batch.shape[0] == action_batch.shape[0] == next_state_batch.shape[0]

    dynamics_model.compute_normalization_stats(
        state_batch=state_batch,
        action_batch=action_batch,
        next_state_batch=next_state_batch,
    )

    parameters = list(dynamics_model.parameters())
    if importance_head is not None:
        parameters += list(importance_head.parameters())

    optimizer = torch.optim.Adam(
        params=parameters, lr=learning_rate, weight_decay=weight_decay
    )

    n_train = state_batch.shape[0]
    steps_per_epoch = max(n_train // batch_size, 1)

    for epoch in tqdm(range(epoch_count)):
        perm = torch.randperm(n=n_train, device=model_device)
        epoch_losses: list[DynamicsLoss] = []
        epoch_gradient_norms: list[torch.Tensor] = []
        epoch_skipped_steps = 0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            optimizer.zero_grad()
            loss = loss_function(
                dynamics_model=dynamics_model,
                state_batch=state_batch[idx],
                action_batch=action_batch[idx],
                next_state_batch=next_state_batch[idx],
            )
            loss.total.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters=parameters, max_norm=1.0
            )

            if torch.isfinite(loss.total) and torch.isfinite(gradient_norm):
                optimizer.step()
            else:
                epoch_skipped_steps += 1

            epoch_losses.append(loss)
            epoch_gradient_norms.append(gradient_norm)

        epoch_mean_loss = _mean_loss(losses=epoch_losses)
        gradient_norms_tensor = torch.stack(epoch_gradient_norms)

        run.log(
            {
                "loss/total": epoch_mean_loss.total.item(),
                "loss/prediction": epoch_mean_loss.prediction.item(),
                "loss/lipschitz": epoch_mean_loss.lipschitz_regularization.item(),
                "gradient_norm/mean": gradient_norms_tensor.mean().item(),
                "gradient_norm/max": gradient_norms_tensor.max().item(),
                "skipped_steps": epoch_skipped_steps,
            },
            step=epoch,
        )


def _mean_loss(losses: list[DynamicsLoss]) -> DynamicsLoss:
    return DynamicsLoss(
        total=torch.stack(tensors=[loss.total.detach() for loss in losses]).mean(),
        prediction=torch.stack(
            tensors=[loss.prediction.detach() for loss in losses]
        ).mean(),
        lipschitz_regularization=torch.stack(
            tensors=[loss.lipschitz_regularization.detach() for loss in losses]
        ).mean(),
    )


def concatenate_episodes(
    episodes: list[Episode],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    states = np.concatenate([episode[0] for episode in episodes], axis=0)
    actions = np.concatenate([episode[1] for episode in episodes], axis=0)
    next_states = np.concatenate([episode[2] for episode in episodes], axis=0)
    return states, actions, next_states


def save_checkpoint(
    dynamics_model: DynamicsModel,
    importance_head: torch.nn.Module | None,
    output_directory_path: Path,
    run: Run,
) -> None:
    dynamics_checkpoint_path = output_directory_path / "dynamics.pt"
    save_dynamics_checkpoint(
        dynamics_model=dynamics_model, path=dynamics_checkpoint_path
    )

    importance_head_path: Path | None = None
    if importance_head is not None:
        importance_head_path = output_directory_path / "importance_head.pt"
        torch.save(obj=importance_head.state_dict(), f=importance_head_path)

    model_artifact = wandb.Artifact(
        name="dynamics_model",
        type="model",
        metadata={
            "state_dimensionality": dynamics_model.state_dimensionality,
            "action_dimensionality": dynamics_model.action_dimensionality,
            "hidden_dimensionalities": list(dynamics_model.hidden_dimensionalities),
            "has_importance_head": importance_head is not None,
        },
    )
    model_artifact.add_file(
        local_path=str(dynamics_checkpoint_path), name="dynamics.pt"
    )
    if importance_head_path is not None:
        model_artifact.add_file(
            local_path=str(importance_head_path), name="importance_head.pt"
        )
    run.log_artifact(artifact_or_path=model_artifact)


def save_dynamics_checkpoint(dynamics_model: DynamicsModel, path: Path) -> None:
    torch.save(
        obj={
            "state_dict": dynamics_model.state_dict(),
            "state_dimensionality": dynamics_model.state_dimensionality,
            "action_dimensionality": dynamics_model.action_dimensionality,
            "hidden_size": dynamics_model.hidden_dimensionalities,
        },
        f=path,
    )


def load_dynamics_checkpoint(path: Path, device: torch.device) -> DynamicsModel:
    checkpoint = torch.load(f=path, map_location=device, weights_only=False)
    dynamics_model = DynamicsModel(
        state_dimensionality=checkpoint["state_dimensionality"],
        action_dimensionality=checkpoint["action_dimensionality"],
        hidden_dimensionalities=tuple(checkpoint["hidden_size"]),
    )
    dynamics_model.load_state_dict(checkpoint["state_dict"])
    return dynamics_model.to(device=device).eval()


def save_evaluation_summary(
    expert_holdout_rollout_errors: dict[int, tuple[float, int]],
    random_rollout_errors: dict[int, tuple[float, int]],
    run: Run,
) -> None:

    horizon_lengths = (
        expert_holdout_rollout_errors.keys() | random_rollout_errors.keys()
    )

    for horizon_length in sorted(horizon_lengths):
        expert_mean, expert_count = expert_holdout_rollout_errors[horizon_length]
        if expert_count > 0:
            run.summary[f"rollout/expert/h_{horizon_length}"] = float(expert_mean)
        rand_mean, rand_count = random_rollout_errors[horizon_length]
        if rand_count > 0:
            run.summary[f"rollout/random/h_{horizon_length}"] = float(rand_mean)


def save_multi_step_rollout_plot(
    expert_holdout_rollout_errors: dict[int, tuple[float, int]],
    random_rollout_errors: dict[int, tuple[float, int]],
    output_directory_path: Path,
    run: Run,
) -> None:

    rollout_horizons = sorted(
        expert_holdout_rollout_errors.keys() | random_rollout_errors.keys()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    figure, plot_area = plt.subplots()

    expert_xs = [h for h in rollout_horizons if expert_holdout_rollout_errors[h][1] > 0]
    expert_ys = [expert_holdout_rollout_errors[h][0] for h in expert_xs]
    plot_area.plot(expert_xs, expert_ys, marker="o", label="Expert holdout")

    rand_xs = [h for h in rollout_horizons if random_rollout_errors[h][1] > 0]
    rand_ys = [random_rollout_errors[h][0] for h in rand_xs]
    plot_area.plot(rand_xs, rand_ys, marker="s", label="Random rollouts")

    plot_area.set_title(label="Multi-step Autoregressive Rollout Error")
    plot_area.set_xlabel(xlabel="Rollout horizon $k$")
    plot_area.set_ylabel(ylabel="Mean L2 error after $k$ steps")
    plot_area.set_yscale(value="log")
    plot_area.legend()

    plot_file_path = output_directory_path / "multi_step_rollout_error.png"
    figure.savefig(fname=plot_file_path)
    plt.close(fig=figure)

    run.log({"rollout/figure": wandb.Image(str(plot_file_path))})


def save_lipschitz_distribution_plot(
    dynamics_model: DynamicsModel,
    lipschitz_constant_constraint: float,
    expert_holdout_states: npt.NDArray[np.float32],
    expert_holdout_actions: npt.NDArray[np.float32],
    random_rollout_episodes: list[Episode],
    output_directory_path: Path,
    run: Run,
) -> None:
    expert_holdout_coefficients = eval_lipschitz_coeff(
        dynamics_model=dynamics_model,
        states=expert_holdout_states,
        actions=expert_holdout_actions,
        batch_size=1_024,
    )

    (random_rollout_episode_states, random_rollout_episode_actions, _) = (
        concatenate_episodes(episodes=random_rollout_episodes)
    )
    random_rollout_coefficients = eval_lipschitz_coeff(
        dynamics_model=dynamics_model,
        states=random_rollout_episode_states,
        actions=random_rollout_episode_actions,
        batch_size=1_024,
    )

    plot_lipschitz_coefficient_distribution(
        expert_holdout_coefficients=expert_holdout_coefficients,
        random_rollout_coefficients=random_rollout_coefficients,
        lipschitz_constraint=lipschitz_constant_constraint,
        output_directory_path=output_directory_path,
    )

    plot_file_path = output_directory_path / "local_lipschitz_distribution.png"

    run.log({"lipschitz/figure": wandb.Image(str(plot_file_path))})

    run.summary["lipschitz/expert_mean"] = float(np.mean(expert_holdout_coefficients))
    run.summary["lipschitz/expert_p99"] = float(
        np.percentile(expert_holdout_coefficients, 99)
    )
    run.summary["lipschitz/expert_max"] = float(np.max(expert_holdout_coefficients))
    run.summary["lipschitz/expert_violation_rate"] = float(
        np.mean(expert_holdout_coefficients > lipschitz_constant_constraint)
    )
    run.summary["lipschitz/random_mean"] = float(np.mean(random_rollout_coefficients))
    run.summary["lipschitz/random_p99"] = float(
        np.percentile(random_rollout_coefficients, 99)
    )
    run.summary["lipschitz/random_max"] = float(np.max(random_rollout_coefficients))
    run.summary["lipschitz/random_violation_rate"] = float(
        np.mean(random_rollout_coefficients > lipschitz_constant_constraint)
    )


@torch.no_grad()
def eval_lipschitz_coeff(
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],
    actions: npt.NDArray[np.float32],
    batch_size: int,
) -> np.ndarray:
    """Per-sample local Lipschitz coefficient (spectral norm of input Jacobian).

    Evaluation always uses the exact Jacobian, regardless of which estimator
    was used during training, so that the reported numbers are comparable
    across runs that trained with different estimation methods.
    """
    model_device = next(dynamics_model.parameters()).device
    state_batch = torch.from_numpy(states).to(device=model_device)
    action_batch = torch.from_numpy(actions).to(device=model_device)

    state_dimensionality = state_batch.shape[-1]
    model_input_batch = torch.cat([state_batch, action_batch], dim=-1)

    def predict_from_concatenated_input(model_input: torch.Tensor) -> torch.Tensor:
        return dynamics_model(
            model_input[..., :state_dimensionality],
            model_input[..., state_dimensionality:],
        )

    batched_input_jacobian = vmap(jacrev(predict_from_concatenated_input))

    spectral_norm_chunks: list[np.ndarray] = []
    for batch_start in tqdm(range(0, model_input_batch.shape[0], batch_size)):
        jacobians = batched_input_jacobian(
            model_input_batch[batch_start : batch_start + batch_size]
        )
        spectral_norms = torch.linalg.norm(jacobians, ord=2, dim=(-2, -1))
        spectral_norm_chunks.append(spectral_norms.cpu().numpy())

    return np.concatenate(spectral_norm_chunks, axis=0)


@torch.no_grad()
def eval_multi_step_rollout_error(
    dynamics_model: DynamicsModel,
    episodes: list[Episode],
    horizon_lengths: tuple[int, ...],
) -> dict[int, tuple[float, int]]:
    model_device = next(dynamics_model.parameters()).device

    errors_per_horizon: dict[int, list[float]] = {k: [] for k in horizon_lengths}

    for states, actions, next_states in episodes:
        # state_trajectory has L+1 entries: [s_0, s_1, ..., s_{L-1}, s_L]
        state_trajectory = torch.from_numpy(
            np.concatenate([states, next_states[-1:]], axis=0)
        ).to(device=model_device)
        action_sequence = torch.from_numpy(actions).to(device=model_device)
        episode_length = states.shape[0]

        for horizon_length in horizon_lengths:
            valid_starting_indices_count = episode_length - horizon_length + 1
            if valid_starting_indices_count <= 0:
                continue

            rollout_states = state_trajectory[:valid_starting_indices_count]  # (V, D)

            for step in range(horizon_length):
                step_actions = action_sequence[
                    step : step + valid_starting_indices_count
                ]
                rollout_states = dynamics_model(rollout_states, step_actions)

            target_states = state_trajectory[
                horizon_length : horizon_length + valid_starting_indices_count
            ]
            errors = torch.linalg.norm(rollout_states - target_states, dim=-1)
            errors_per_horizon[horizon_length].extend(errors.cpu().numpy().tolist())

    result: dict[int, tuple[float, int]] = {}
    for horizon_length, errors in errors_per_horizon.items():
        if len(errors) == 0:
            result[horizon_length] = (float("nan"), 0)
        else:
            result[horizon_length] = (float(np.mean(errors)), len(errors))
    return result


def plot_lipschitz_coefficient_distribution(
    expert_holdout_coefficients: np.ndarray,
    random_rollout_coefficients: np.ndarray,
    lipschitz_constraint: float,
    output_directory_path: Path,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    figure = plt.figure()
    plot_area = figure.add_subplot()
    plot_area.set_title("Distribution of Local Lipschitz Coefficients")
    plot_area.set_xlabel("Lipschitz Coefficient")
    plot_area.set_ylabel("Density")

    combined = np.concatenate(
        [expert_holdout_coefficients, random_rollout_coefficients]
    )

    min_x_value = np.min(combined)
    max_x_value = np.percentile(combined, q=99.5)

    x_values = np.linspace(start=min_x_value, stop=max_x_value, num=1_000)

    expert_density = gaussian_kde(expert_holdout_coefficients)(x_values)
    random_density = gaussian_kde(random_rollout_coefficients)(x_values)

    plot_area.fill_between(x_values, expert_density, alpha=0.6, label="Expert holdout")
    plot_area.fill_between(x_values, random_density, alpha=0.6, label="Random rollouts")

    plot_area.axvline(
        x=lipschitz_constraint, linestyle="--", color="black", label="Constraint"
    )
    plot_area.legend()
    figure.savefig(fname=output_directory_path / "local_lipschitz_distribution.png")
    plt.close(figure)


if __name__ == "__main__":
    tyro.cli(f=main)
