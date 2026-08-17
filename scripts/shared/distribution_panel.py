import matplotlib
import numpy as np
import wandb

matplotlib.use(backend="Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


def distribution_panel(*, values: np.ndarray, column_name: str) -> wandb.Image:
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.hist(values, bins=_bins(values=values), edgecolor="white")
    axes.set_xlabel(xlabel=column_name)
    if np.issubdtype(values.dtype, np.integer):
        # The bars sit on integers, so the ticks have to as well. Left to
        # itself the axis places them at the bin edges, which are halves.
        axes.xaxis.set_major_locator(locator=MaxNLocator(integer=True))
    # Every value is drawn, and a log count axis keeps the peak from flattening
    # the tail. Cutting the tail off instead would hide the rare extremes,
    # which are the ones worth seeing.
    axes.set_ylabel(ylabel="count")
    axes.set_yscale(value="log")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


def _bins(*, values: np.ndarray) -> list[float] | int:
    if np.issubdtype(values.dtype, np.integer):
        # One bar per integer value, centered on it.
        return np.arange(values.min() - 0.5, values.max() + 1.5).tolist()

    edges = np.linspace(float(values.min()), float(values.max()), num=21)
    return edges.tolist() if np.all(np.diff(edges) > 0.0) else 1
