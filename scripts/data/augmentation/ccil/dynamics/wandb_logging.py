from pathlib import Path

import matplotlib

matplotlib.use(backend="Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from wandb import Run

from scripts.data.augmentation.ccil.dynamics.evaluation import BackwardRolloutResult
from scripts.data.augmentation.ccil.dynamics.losses import LossBreakdown


def log_epoch(
    loss_breakdowns: list[LossBreakdown],
    gradient_norms: list[torch.Tensor],
    epoch_index: int,
    run: Run,
) -> None:
    mean_loss_breakdown = _create_mean_breakdown(breakdowns=loss_breakdowns)

    run.log(
        {
            "loss/total": mean_loss_breakdown.total.item(),
            "loss/forward": mean_loss_breakdown.forward.item(),
            "loss/inverse": mean_loss_breakdown.inverse.item(),
            "gradient_norm": torch.stack(gradient_norms).max().item(),
        },
        step=epoch_index,
    )


def _create_mean_breakdown(breakdowns: list[LossBreakdown]) -> LossBreakdown:
    return LossBreakdown(
        total=torch.stack([breakdown.total for breakdown in breakdowns]).mean(),
        forward=torch.stack([breakdown.forward for breakdown in breakdowns]).mean(),
        inverse=torch.stack([breakdown.inverse for breakdown in breakdowns]).mean(),
    )


def log_rollout_summaries(
    forward_rollout_errors: dict[int, float],
    clean_backward_rollout_results: dict[int, BackwardRolloutResult],
    noisy_backward_rollout_results: dict[int, BackwardRolloutResult],
    objective_horizon_length: int,
    run: Run,
) -> None:
    _log_objective(
        clean_backward_rollout_results=clean_backward_rollout_results,
        objective_horizon_length=objective_horizon_length,
        run=run,
    )
    _log_operating_point_summary(
        forward_rollout_errors=forward_rollout_errors,
        clean_backward_rollout_results=clean_backward_rollout_results,
        noisy_backward_rollout_results=noisy_backward_rollout_results,
        horizon_length=objective_horizon_length,
        run=run,
    )


def _log_objective(
    clean_backward_rollout_results: dict[int, BackwardRolloutResult],
    objective_horizon_length: int,
    run: Run,
) -> None:
    run.summary["objective/horizon_length"] = objective_horizon_length

    clean_backward_rollout_result = clean_backward_rollout_results.get(
        objective_horizon_length
    )

    # NaN if no valid windows at this depth (episodes are too short).
    run.summary["objective/clean_backward_rollout_error"] = (
        float("nan")
        if clean_backward_rollout_result is None
        else clean_backward_rollout_result.error
    )


def _log_operating_point_summary(
    forward_rollout_errors: dict[int, float],
    clean_backward_rollout_results: dict[int, BackwardRolloutResult],
    noisy_backward_rollout_results: dict[int, BackwardRolloutResult],
    horizon_length: int,
    run: Run,
) -> None:
    prefix = f"rollout/h_{horizon_length}"

    clean_backward_rollout_result = clean_backward_rollout_results.get(horizon_length)
    if clean_backward_rollout_result is not None:
        run.summary[f"{prefix}/backward_error_clean"] = (
            clean_backward_rollout_result.error
        )
        run.summary[f"{prefix}/backward_newton_step_success_rate_clean"] = (
            clean_backward_rollout_result.newton_step_success_rate
        )

    noisy_backward_rollout_result = noisy_backward_rollout_results.get(horizon_length)
    if noisy_backward_rollout_result is not None:
        run.summary[f"{prefix}/backward_error_noisy"] = (
            noisy_backward_rollout_result.error
        )
        run.summary[f"{prefix}/backward_newton_step_success_rate_noisy"] = (
            noisy_backward_rollout_result.newton_step_success_rate
        )

    forward_rollout_error = forward_rollout_errors.get(horizon_length)
    if forward_rollout_error is not None:
        run.summary[f"{prefix}/forward_error"] = forward_rollout_error


def log_local_lipschitz_p99_and_violation_rate(
    local_lipschitz_constants: np.ndarray,
    lipschitz_constant_upper_bound: float | None,
    run: Run,
) -> None:
    run.summary["lipschitz/p99"] = np.percentile(local_lipschitz_constants, q=99).item()
    run.summary["lipschitz/violation_rate"] = (
        float("nan")
        if lipschitz_constant_upper_bound is None
        else np.mean(local_lipschitz_constants > lipschitz_constant_upper_bound).item()
    )


def _apply_style() -> None:
    # noinspection SpellCheckingInspection
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300


def log_rollout_error_plot(
    forward_results: dict[int, float],
    clean_backward_rollout_results: dict[int, BackwardRolloutResult],
    noisy_backward_rollout_results: dict[int, BackwardRolloutResult],
    run: Run,
) -> None:
    _apply_style()

    figure = plt.figure()
    plot_area = figure.add_subplot()

    def _plot_forward(results: dict[int, float], marker: str) -> None:
        if not results:
            return

        sorted_horizon_lengths = sorted(results.keys())
        errors = np.asarray(
            [results[horizon_length] for horizon_length in sorted_horizon_lengths]
        )

        plot_area.plot(sorted_horizon_lengths, errors, marker=marker, label="Forward")

    def _plot_backward(
        results: dict[int, BackwardRolloutResult], marker: str, label: str
    ) -> None:
        if not results:
            return

        sorted_horizon_lengths = sorted(results.keys())
        errors = np.asarray(
            [results[horizon_length].error for horizon_length in sorted_horizon_lengths]
        )

        plot_area.plot(
            sorted_horizon_lengths, errors, marker=marker, label=f"Backward ({label})"
        )

    _plot_forward(forward_results, marker="o")

    _plot_backward(clean_backward_rollout_results, marker="s", label="clean")

    _plot_backward(noisy_backward_rollout_results, marker="^", label="noisy")

    plot_area.set_xlabel("Horizon length")
    plot_area.set_ylabel("Error")
    plot_area.set_title("Error vs Horizon length")
    plot_area.legend()

    save_path = Path(run.dir) / "rollout_error_vs_horizon.png"

    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    run.save(save_path, base_path=save_path.parent, policy="now")


def log_local_lipschitz_constant_distribution_plot(
    local_lipschitz_constants: np.ndarray,
    lipschitz_constant_upper_bound: float | None,
    run: Run,
) -> None:
    _apply_style()

    figure = plt.figure()
    plot_area = figure.add_subplot()
    plot_area.set_title("Distribution of Local Lipschitz Constants")
    plot_area.set_xlabel("Local Lipschitz Constant")
    plot_area.set_ylabel("Density")

    min_x_value = float(np.min(local_lipschitz_constants))
    max_x_value = float(np.percentile(local_lipschitz_constants, q=99.5))

    x_values = np.linspace(start=min_x_value, stop=max_x_value, num=1_000)
    density = gaussian_kde(local_lipschitz_constants)(x_values)

    plot_area.fill_between(x_values, density, alpha=0.6)
    if lipschitz_constant_upper_bound is not None:
        plot_area.axvline(
            x=lipschitz_constant_upper_bound,
            linestyle="--",
            color="black",
            label="Upper Bound",
        )
    plot_area.legend()

    save_path = Path(run.dir) / "lipschitz_distribution.png"

    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    run.save(save_path, base_path=save_path.parent, policy="now")
