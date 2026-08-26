from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    action_noise_range_fraction: float = 0.1

    # Ceiling on the replay error, in multiples of the median expert step
    # distance. The whole candidate chain is replayed forward with every
    # ensemble member, and what is measured is how far their endpoints land
    # from the anchor — how badly the model misses what the synthetic episode
    # claims: that these actions lead to the anchor.
    replay_error_limit_step_multiple: float = 6.0

    # Label each synthetic state with GPI's geometric action (pull toward the
    # nearest demonstration and follow it) instead of the anchor action. The
    # backward rollout still shapes which states are invented; only their label
    # changes. The two weights are GPI's progression and attraction weights.
    use_gpi_labels: bool = False
    gpi_progression_weight: float = 1.0
    gpi_attraction_weight: float = 1.0
