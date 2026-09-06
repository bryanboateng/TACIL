import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt

matplotlib.use(backend="Agg")


def create_nearest_recorded_state_plot(
    *,
    synthetic_state_distances: np.ndarray,  # (synthetic_count,)
    synthetic_action_distances: np.ndarray,  # (synthetic_count,)
    recorded_state_distances: np.ndarray,  # (recorded_count,)
    recorded_action_distances: np.ndarray,  # (recorded_count,)
) -> wandb.Image:
    """Where the invented steps sit relative to the recorded ones. Left of the
    band and below it is a copy of the data, above it is a second label for a
    state the data already covers, and right of it is the model extrapolating.
    """
    figure, axes = plt.subplots(figsize=(6, 5))

    axes.add_patch(
        plt.Rectangle(
            xy=(
                _percentile(recorded_state_distances, 10),
                _percentile(recorded_action_distances, 10),
            ),
            width=_percentile(recorded_state_distances, 90)
            - _percentile(recorded_state_distances, 10),
            height=_percentile(recorded_action_distances, 90)
            - _percentile(recorded_action_distances, 10),
            facecolor="#c8c8c8",
            edgecolor="#8c8c8c",
            alpha=0.55,
            zorder=1,
            label="recorded, p10 to p90",
        )
    )
    for draw_line, values in (
        (axes.axvline, recorded_state_distances),
        (axes.axhline, recorded_action_distances),
    ):
        draw_line(_percentile(values, 50), color="#6e6e6e", linewidth=1, zorder=2)

    drawn = _draw_sample(size=synthetic_state_distances.shape[0])
    axes.scatter(
        synthetic_state_distances[drawn],
        synthetic_action_distances[drawn],
        s=4,
        alpha=0.3,
        linewidths=0,
        color="#1f77b4",
        zorder=3,
        label="synthetic",
    )

    axes.set_xlabel(xlabel="distance to nearest recorded state, in expert steps")
    axes.set_ylabel(ylabel="distance to that state's action")

    # A handful of far outliers would otherwise squeeze the cloud into a
    # corner, and the cloud is what the plot is for.
    axes.set_xlim(left=0.0, right=_percentile(synthetic_state_distances, 99))
    axes.set_ylim(bottom=0.0, top=_percentile(synthetic_action_distances, 99))
    axes.legend(markerscale=3, loc="upper left")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


_DRAWN_POINT_COUNT = 4_000


def _draw_sample(*, size: int) -> np.ndarray:
    if size <= _DRAWN_POINT_COUNT:
        return np.arange(size)
    return np.random.default_rng(seed=0).choice(
        size, size=_DRAWN_POINT_COUNT, replace=False
    )


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(a=values, q=percentile))
