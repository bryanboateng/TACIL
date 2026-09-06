import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.use(backend="Agg")

plt.style.use(style="seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


def create_distribution_plot(*, values: np.ndarray, xlabel: str) -> wandb.Image:
    figure, plot_area = plt.subplots(figsize=(6, 4))

    plot_area.hist(x=values, bins=_choose_bins(values=values), edgecolor="white")

    plot_area.set_xlabel(xlabel=xlabel)

    if np.issubdtype(values.dtype, np.integer):
        # The bars sit on integers, so the ticks have to as well. Left to
        # itself the axis places them at the bin edges, which are halves.
        plot_area.xaxis.set_major_locator(locator=MaxNLocator(integer=True))

    plot_area.set_ylabel(ylabel="count")
    plot_area.set_yscale(value="log")
    figure.tight_layout()

    image = wandb.Image(figure)
    plt.close(fig=figure)
    return image


# ---- Private --------------------------------------------


def _choose_bins(*, values: np.ndarray) -> list[float] | int:
    if np.issubdtype(values.dtype, np.integer):
        return np.arange(values.min() - 0.5, values.max() + 1.5).tolist()

    minimum = float(values.min())
    maximum = float(values.max())

    if minimum == maximum:
        return 1

    return np.linspace(minimum, maximum, num=21).tolist()
