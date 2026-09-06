import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt

matplotlib.use(backend="Agg")


def create_nearest_recorded_state_panel(
    *,
    synthetic_state_distances: np.ndarray,  # (synthetic_count,)
    synthetic_action_distances: np.ndarray,  # (synthetic_count,)
    recorded_state_distances: np.ndarray,  # (recorded_count,)
    recorded_action_distances: np.ndarray,  # (recorded_count,)
) -> wandb.Image:
    """Where the invented steps sit relative to the recorded ones. Bottom left
    is a copy of the data, top left is a second label for a state the data
    already covers, and far right is the model extrapolating.
    """
    drawn_point_count = 4_000
    figure, axes = plt.subplots(figsize=(6, 5))

    for state_distances, action_distances, label, color in (
        (recorded_state_distances, recorded_action_distances, "recorded", "#888888"),
        (synthetic_state_distances, synthetic_action_distances, "synthetic", "#1f77b4"),
    ):
        drawn = _draw_sample(count=drawn_point_count, size=state_distances.shape[0])
        axes.scatter(
            state_distances[drawn],
            action_distances[drawn],
            s=4,
            alpha=0.25,
            linewidths=0,
            color=color,
            label=label,
        )

    axes.axhline(
        y=float(np.median(recorded_action_distances)), color="#888888", linewidth=1
    )
    axes.set_xlabel(xlabel="distance to nearest recorded state, in expert steps")
    axes.set_ylabel(ylabel="distance to that state's action")

    # A handful of far outliers would otherwise squeeze both clouds into a
    # corner, and the clouds are what the panel is for.
    axes.set_xlim(left=0.0, right=_upper_limit(synthetic_state_distances))
    axes.set_ylim(bottom=0.0, top=_upper_limit(synthetic_action_distances))
    axes.legend(markerscale=3)
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


def _draw_sample(*, count: int, size: int) -> np.ndarray:
    if size <= count:
        return np.arange(size)
    return np.random.default_rng(seed=0).choice(size, size=count, replace=False)


def _upper_limit(values: np.ndarray) -> float:
    return float(np.percentile(a=values, q=99)) or 1.0
