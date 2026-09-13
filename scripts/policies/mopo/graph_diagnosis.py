"""Why the learning step is not replayed as a graph.

Two questions. Which tensors reachable from the policy do not sit on the
device, and what does the compiler itself say when asked loudly.

    CUDA_VISIBLE_DEVICES=<n> uv run python -m scripts.policies.mopo.graph_diagnosis
"""

import copy
import io
from typing import cast
import logging
import torch

from scripts.policies.mopo.buffer import Batch
from scripts.policies.mopo.dynamics import Dynamics
from scripts.policies.mopo.policy import EnsembleCritic, GaussianActor, MopoPolicy

STATE_DIMENSIONALITY, ACTION_DIMENSIONALITY, BATCH_SIZE = 17, 6, 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device {DEVICE}")
if DEVICE.type == "cuda":
    capability = torch.cuda.get_device_capability(DEVICE)
    print(
        f"gpu    {torch.cuda.get_device_name(DEVICE)}, capability {capability[0]}.{capability[1]}"
    )
print()

torch.manual_seed(0)
actor = GaussianActor(
    state_dimensionality=STATE_DIMENSIONALITY,
    action_dimensionality=ACTION_DIMENSIONALITY,
    hidden_layer_dimensionalities=(256, 256),
).to(device=DEVICE)
critic = EnsembleCritic(
    state_dimensionality=STATE_DIMENSIONALITY,
    action_dimensionality=ACTION_DIMENSIONALITY,
    hidden_layer_dimensionalities=(256, 256),
).to(device=DEVICE)
log_temperature = torch.zeros(1, requires_grad=True, device=DEVICE)
optimizers = {
    "actor": torch.optim.Adam(actor.parameters(), lr=1e-4, fused=True),
    "temperature": torch.optim.Adam([log_temperature], lr=1e-4, fused=True),
    "critic": torch.optim.Adam(critic.parameters(), lr=3e-4, fused=True),
}
policy = MopoPolicy(
    actor=actor,
    critic=critic,
    target_critic=copy.deepcopy(critic).eval(),
    actor_optimizer=optimizers["actor"],
    temperature_optimizer=optimizers["temperature"],
    critic_optimizer=optimizers["critic"],
    log_temperature=log_temperature,
    target_entropy=-float(ACTION_DIMENSIONALITY),
    # Never reached: only the learning step is exercised here.
    dynamics=cast(Dynamics, None),
    discount_factor=0.99,
    target_update_rate=0.005,
    compile_learning_step=False,
    device=DEVICE,
)
batch = Batch(
    states=torch.randn(BATCH_SIZE, STATE_DIMENSIONALITY, device=DEVICE),
    actions=torch.randn(BATCH_SIZE, ACTION_DIMENSIONALITY, device=DEVICE),
    next_states=torch.randn(BATCH_SIZE, STATE_DIMENSIONALITY, device=DEVICE),
    rewards=torch.randn(BATCH_SIZE, 1, device=DEVICE),
    terminations=torch.zeros(BATCH_SIZE, 1, device=DEVICE),
)

# One plain step so the optimizers build their state.
policy.learn(batch=batch)

print("--- tensors not on the device ---")
strays = []
for name, module in (
    ("actor", actor),
    ("critic", critic),
    ("target_critic", policy.target_critic),
):
    for parameter_name, tensor in list(module.named_parameters()) + list(
        module.named_buffers()
    ):
        if tensor.device.type != DEVICE.type:
            strays.append(
                f"{name}.{parameter_name}  {tensor.device}  {tuple(tensor.shape)}"
            )
for name, tensor in (
    ("log_temperature", log_temperature),
    ("_statistic_sums", policy._statistic_sums),
):
    if tensor.device.type != DEVICE.type:
        strays.append(f"{name}  {tensor.device}  {tuple(tensor.shape)}")
for optimizer_name, optimizer in optimizers.items():
    for index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type != DEVICE.type:
                strays.append(
                    f"{optimizer_name}_optimizer.state[{index}][{key}]  {value.device}  {tuple(value.shape)}"
                )
print("\n".join(f"  {one}" for one in strays) if strays else "  none")

print()
print("--- what the compiler says ---")
logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.DEBUG)
buffer = io.StringIO()
handler = logging.StreamHandler(buffer)
for logger_name in (
    "torch._inductor.cudagraph_trees",
    "torch._inductor.compile_fx",
    "torch._dynamo",
):
    logging.getLogger(logger_name).addHandler(handler)

ITERATION_COUNT = 40

compiled = torch.compile(policy._run_learn_step, mode="reduce-overhead")
for _ in range(ITERATION_COUNT):
    torch.compiler.cudagraph_mark_step_begin()
    policy._statistic_sums += compiled(batch)
if DEVICE.type == "cuda":
    torch.cuda.synchronize()

# Which state each fragment ended up in, counted over the whole run.
counts = {
    "recorded": 0,
    "warmed up": 0,
    "ran eager, ancestor warming": 0,
    "replayed": 0,
}
for line in buffer.getvalue().splitlines():
    if "recording cudagraph tree" in line:
        counts["recorded"] += 1
    elif "Running warmup of function" in line:
        counts["warmed up"] += 1
    elif "because ancestor needed to warm up" in line:
        counts["ran eager, ancestor warming"] += 1
    elif "replay" in line.lower():
        counts["replayed"] += 1

print(f"  over {ITERATION_COUNT} iterations:")
for label, count in counts.items():
    print(f"    {label:30s} {count}")
print()
print(
    "  Fragments are replayed, so the bundling does something."
    if counts["replayed"]
    else "  Nothing was ever replayed. The bundling costs and gives nothing."
)
