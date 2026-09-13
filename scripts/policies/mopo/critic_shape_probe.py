"""Stacked critics against two separate ones, with and without bundling.

Soft actor critic needs two critics. This repository holds them in one stack
so that a single batched product covers both, which costs a parameter list,
a batched product and an extra axis in every shape. Whether that still pays
depends on how much the compiler already fuses on its own, so it is measured
rather than assumed.

    CUDA_VISIBLE_DEVICES=<n> uv run python -m scripts.policies.mopo.critic_shape_probe
"""

import copy
import statistics
import time
from typing import cast

import torch

from scripts.policies.mopo.buffer import Batch
from scripts.policies.mopo.dynamics import Dynamics
from scripts.policies.mopo.policy import (
    EnsembleCritic,
    GaussianActor,
    MopoPolicy,
    _create_backbone,
)

STATE_DIMENSIONALITY, ACTION_DIMENSIONALITY, BATCH_SIZE = 17, 6, 256
HIDDEN = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeparateCritics(torch.nn.Module):
    """What the reference has: two plain critics, stacked only at the end."""

    def __init__(
        self,
        *,
        state_dimensionality,
        action_dimensionality,
        hidden_layer_dimensionalities,
    ):
        super().__init__()
        self.members = torch.nn.ModuleList(
            torch.nn.Sequential(
                _create_backbone(
                    input_dimensionality=state_dimensionality + action_dimensionality,
                    hidden_layer_dimensionalities=hidden_layer_dimensionalities,
                ),
                torch.nn.Linear(hidden_layer_dimensionalities[-1], 1),
            )
            for _ in range(2)
        )

    def forward(self, input_batch):
        return torch.stack([member(input_batch) for member in self.members])


def build(critic_class, compile_learning_step):
    torch.manual_seed(0)
    actor = GaussianActor(
        state_dimensionality=STATE_DIMENSIONALITY,
        action_dimensionality=ACTION_DIMENSIONALITY,
        hidden_layer_dimensionalities=HIDDEN,
    ).to(device=DEVICE)
    critic = critic_class(
        state_dimensionality=STATE_DIMENSIONALITY,
        action_dimensionality=ACTION_DIMENSIONALITY,
        hidden_layer_dimensionalities=HIDDEN,
    ).to(device=DEVICE)
    log_temperature = torch.zeros(1, requires_grad=True, device=DEVICE)
    return MopoPolicy(
        actor=actor,
        # SeparateCritics stands in for the stacked one on purpose: it answers
        # the same call and returns the same shape.
        critic=cast(EnsembleCritic, critic),
        target_critic=cast(EnsembleCritic, copy.deepcopy(critic).eval()),
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-4, fused=True),
        temperature_optimizer=torch.optim.Adam([log_temperature], lr=1e-4, fused=True),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=3e-4, fused=True),
        log_temperature=log_temperature,
        target_entropy=-float(ACTION_DIMENSIONALITY),
        # Never reached: the probe only ever calls the learning step.
        dynamics=cast(Dynamics, None),
        discount_factor=0.99,
        target_update_rate=0.005,
        compile_learning_step=compile_learning_step,
        device=DEVICE,
    )


print(f"device {DEVICE}")
if DEVICE.type == "cuda":
    capability = torch.cuda.get_device_capability(DEVICE)
    print(
        f"gpu    {torch.cuda.get_device_name(DEVICE)}, "
        f"capability {capability[0]}.{capability[1]}"
    )
print()

batch = Batch(
    states=torch.randn(BATCH_SIZE, STATE_DIMENSIONALITY, device=DEVICE),
    actions=torch.randn(BATCH_SIZE, ACTION_DIMENSIONALITY, device=DEVICE),
    next_states=torch.randn(BATCH_SIZE, STATE_DIMENSIONALITY, device=DEVICE),
    rewards=torch.randn(BATCH_SIZE, 1, device=DEVICE),
    terminations=torch.zeros(BATCH_SIZE, 1, device=DEVICE),
)

for compiled in (False, True):
    policies = {
        "gestapelt": build(EnsembleCritic, compiled),
        "getrennt": build(SeparateCritics, compiled),
    }
    warmup = 150 if compiled else 40
    for policy in policies.values():
        for _ in range(warmup):
            policy.learn(batch=batch)
        policy.take_average_statistics(step_count=warmup)

    runs = {label: [] for label in policies}
    for _ in range(5):
        for label, policy in policies.items():
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(120):
                policy.learn(batch=batch)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            runs[label].append((time.perf_counter() - start) / 120 * 1000)
            policy.take_average_statistics(step_count=120)

    stacked = statistics.median(runs["gestapelt"])
    separate = statistics.median(runs["getrennt"])
    tag = "bundled" if compiled else "plain"
    print(
        f"  {tag:9s} stacked {stacked:6.3f} ms   separate {separate:6.3f} ms"
        f"   -> stacked is {separate / stacked:.2f}x faster"
    )
