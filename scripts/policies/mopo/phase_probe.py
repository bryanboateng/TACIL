"""Times the three phases of a MOPO run on whatever machine it runs on.

A run trains the dynamics models once, then repeats a rollout through them
followed by a thousand learning steps. Which of the three dominates decides
where any further work belongs. A full run reports the last two to wandb
anyway; this answers the same question in minutes rather than hours, and
covers the model training as well.

    CUDA_VISIBLE_DEVICES=<n> uv run python -m scripts.policies.mopo.phase_probe
"""

import time

import numpy as np
import torch

from scripts.policies.mopo import config as mopo_config
from scripts.policies.mopo.buffer import ReplayBuffer, concatenate_batches
from scripts.policies.mopo.config import DynamicsConfig
from scripts.policies.mopo.dynamics import (
    Dynamics,
    DynamicsModelEnsemble,
    InputNormalizer,
)
from scripts.policies.mopo.dynamics_training import _train_for_one_epoch
from scripts.policies.mopo.replication.dataset import load_dataset
from scripts.policies.mopo.replication.tasks import ALL_TASKS
from scripts.policies.mopo.replication.termination import get_termination_function
from scripts.policies.mopo.train import _create_policy
from scripts.policies.mopo.transitions import Transitions

TASK_NAME = "walker2d-medium-v2"
MEASURED_EPOCH_COUNT = 3
LEARNING_STEP_COUNT = 1000
WARMUP_STEP_COUNT = 200
EPOCH_COUNT = 3000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        print(
            f"gpu    {torch.cuda.get_device_name(device)}, capability {capability[0]}.{capability[1]}"
        )
    print(f"threads {torch.get_num_threads()}")

    task = ALL_TASKS[TASK_NAME]
    dataset = load_dataset(task=task)
    print(
        f"task   {task.name}: {len(dataset.states)} transitions, h={task.rollout_length}\n"
    )

    state_dimensionality = dataset.state_dimensionality
    action_dimensionality = dataset.action_dimensionality

    config = mopo_config.Config(
        dynamics=mopo_config.DynamicsConfig(),
        rollout=mopo_config.RolloutConfig(
            length=task.rollout_length, penalty_coefficient=task.penalty_coefficient
        ),
        optimization=mopo_config.OptimizationConfig(),
        validation=mopo_config.ValidationConfig(),
    )

    transitions = Transitions(
        states=torch.from_numpy(dataset.states).to(device=device),
        actions=torch.from_numpy(dataset.actions).to(device=device),
        next_states=torch.from_numpy(dataset.next_states).to(device=device),
        rewards=torch.from_numpy(dataset.rewards).to(device=device),
        terminations=torch.from_numpy(dataset.terminations).to(device=device),
    )
    inputs = torch.cat(tensors=[transitions.states, transitions.actions], dim=-1)
    targets = torch.cat(
        tensors=[transitions.next_states - transitions.states, transitions.rewards],
        dim=-1,
    )
    ensemble = (
        DynamicsModelEnsemble(
            config=config.dynamics,
            state_dimensionality=state_dimensionality,
            action_dimensionality=action_dimensionality,
        )
        .to(device)
        .eval()
    )
    dynamics = Dynamics(
        model_ensemble=ensemble,
        optimizer=torch.optim.Adam(ensemble.parameters()),
        input_normalizer=InputNormalizer(value_batch=inputs),
        termination_function=get_termination_function(robot=task.robot),
        penalty_coefficient=task.penalty_coefficient,
        device=device,
    )

    dynamics_seconds = _measure_dynamics_epoch(
        config=config.dynamics,
        input_batch=dynamics.input_normalizer.normalize(value_batch=inputs),
        target_batch=targets,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )
    print(f"one dynamics epoch: {dynamics_seconds:.2f} s")

    policy = _create_policy(
        config=config,
        dynamics=dynamics,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )
    policy.train()

    real_buffer = ReplayBuffer(
        capacity=len(dataset.states),
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )
    real_buffer.add(transitions=transitions)
    imagined_buffer = ReplayBuffer(
        capacity=config.rollout.start_state_count
        * config.rollout.length
        * config.rollout.retained_epoch_count,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )

    def roll_out():
        rollout, statistics = policy.imagine_rollout(
            start_state_batch=real_buffer.sample_states(
                count=config.rollout.start_state_count
            ),
            length=config.rollout.length,
        )
        imagined_buffer.add(transitions=rollout)
        return statistics

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    statistics = roll_out()
    print(
        f"first rollout: {statistics.transition_count} transitions, "
        f"reward {statistics.reward_mean:.3f}, penalty {statistics.penalty_mean:.3f}"
    )

    real_batch_size = int(
        config.optimization.batch_size * config.optimization.real_data_ratio
    )
    imagined_batch_size = config.optimization.batch_size - real_batch_size

    def learn_once():
        policy.learn(
            batch=concatenate_batches(
                first=real_buffer.sample(count=real_batch_size),
                second=imagined_buffer.sample(count=imagined_batch_size),
            )
        )

    print("warming up (the first bundled steps are compiled) ...")
    for _ in range(WARMUP_STEP_COUNT):
        learn_once()
    synchronize()
    policy.take_average_statistics(step_count=WARMUP_STEP_COUNT)

    print()
    rollout_seconds = []
    learning_seconds = []
    for epoch in range(MEASURED_EPOCH_COUNT):
        synchronize()
        start = time.perf_counter()
        roll_out()
        synchronize()
        rollout_seconds.append(time.perf_counter() - start)

        start = time.perf_counter()
        for _ in range(LEARNING_STEP_COUNT):
            learn_once()
        synchronize()
        learning_seconds.append(time.perf_counter() - start)
        policy.take_average_statistics(step_count=LEARNING_STEP_COUNT)
        print(
            f"  epoch {epoch + 1}: rollout {rollout_seconds[-1]:6.2f} s   "
            f"learning {learning_seconds[-1]:6.2f} s"
        )

    rollout_mean = float(np.mean(rollout_seconds))
    learning_mean = float(np.mean(learning_seconds))
    total = rollout_mean + learning_mean

    print()
    print(f"{'dynamics epoch (once)':24s}{dynamics_seconds:8.2f} s")
    print(
        f"{'rollout per epoch':24s}{rollout_mean:8.2f} s   {rollout_mean / total * 100:4.1f} %"
    )
    print(
        f"{'learning per epoch':24s}{learning_mean:8.2f} s   {learning_mean / total * 100:4.1f} %"
    )
    print(f"{'together':24s}{total:8.2f} s")
    print()
    print(f"{EPOCH_COUNT} epochs would take {total * EPOCH_COUNT / 3600:.1f} h")


def _measure_dynamics_epoch(
    *,
    config: DynamicsConfig,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    state_dimensionality: int,
    action_dimensionality: int,
    device: torch.device,
) -> float:
    """One pass of dynamics training. A run does dozens before it starts."""
    generator = torch.Generator(device=device).manual_seed(0)
    member_indices = torch.randint(
        low=0,
        high=len(input_batch),
        size=(config.ensemble_size, len(input_batch)),
        generator=generator,
        device=device,
    )
    ensemble = DynamicsModelEnsemble(
        config=config,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
    ).to(device=device)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=config.learning_rate)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _train_for_one_epoch(
        config=config,
        model_ensemble=ensemble,
        optimizer=optimizer,
        input_batch=input_batch,
        target_batch=target_batch,
        member_indices=member_indices,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


if __name__ == "__main__":
    main()
