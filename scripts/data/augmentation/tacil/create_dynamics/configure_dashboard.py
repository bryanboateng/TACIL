"""Curated W&B saved view for the TACIL dynamics ensemble.

``ensure_dashboard_view`` is called once per run from the training entrypoint. It upserts
a single saved view named "Ensemble dashboard" in which each mean/standard-deviation band
is one panel (the mean plus its two edges on one chart) rather than three separate
auto-generated panels. It is idempotent: it always targets the same view via a fixed
internal name, so repeated runs update it in place instead of creating duplicates, and no
one has to build panels by hand.

The distribution histograms (``lipschitz/distribution_*``) are still logged and visible in
the project's default view; the workspace API has no histogram panel type, so they cannot
be placed in this curated view.
"""

import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws

from scripts.shared.log import create_logger

logger = create_logger()

_VIEW_NAME = "Ensemble dashboard"
# A fixed internal view name makes save() upsert the same saved view every run instead of
# creating a new one; it must keep the "nw-<slug>-v" shape the package round-trips.
_INTERNAL_VIEW_NAME = "nw-tacil-ensemble-dashboard-v"


def _band_panel(*, title: str, metric_base: str) -> wr.LinePlot:
    """One panel holding a band's mean line and its two standard-deviation edges."""
    mean_series = f"{metric_base}_mean"
    lower_series = f"{metric_base}_band_lower"
    upper_series = f"{metric_base}_band_upper"
    return wr.LinePlot(
        title=title,
        y=[mean_series, lower_series, upper_series],
        line_titles={
            mean_series: "mean",
            lower_series: "-1 std",
            upper_series: "+1 std",
        },
        line_widths={mean_series: 2.0, lower_series: 1.0, upper_series: 1.0},
    )


def _build_workspace(*, entity: str, project: str) -> ws.Workspace:
    return ws.Workspace(
        entity=entity,
        project=project,
        name=_VIEW_NAME,
        auto_generate_panels=False,
        sections=[
            ws.Section(
                name="Loss",
                is_open=True,
                panels=[
                    wr.LinePlot(title="training loss", y=["loss/training_mean"]),
                    _band_panel(
                        title="validation loss", metric_base="loss/validation"
                    ),
                ],
            ),
            ws.Section(
                name="Lipschitz",
                is_open=True,
                panels=[
                    _band_panel(
                        title="mean (training)", metric_base="lipschitz/mean_training"
                    ),
                    _band_panel(
                        title="mean (validation)",
                        metric_base="lipschitz/mean_validation",
                    ),
                    _band_panel(
                        title="p99 (training)", metric_base="lipschitz/p99_training"
                    ),
                    _band_panel(
                        title="p99 (validation)",
                        metric_base="lipschitz/p99_validation",
                    ),
                ],
            ),
            ws.Section(
                name="Health",
                is_open=True,
                panels=[
                    wr.LinePlot(title="gradient norm (max)", y=["gradient_norm_max"])
                ],
            ),
        ],
    )


def ensure_dashboard_view(*, entity: str, project: str) -> None:
    """Upsert the curated saved view for this project.

    Best-effort: configuring the dashboard must never break a training run, so any
    failure (offline mode, API change, missing permissions) is logged and swallowed.
    """
    try:
        workspace = _build_workspace(entity=entity, project=project)
        # Pin the internal name so repeated runs update this one view rather than
        # spawning a new duplicate view each time.
        object.__setattr__(workspace, "_internal_name", _INTERNAL_VIEW_NAME)
        workspace.save()
        logger.info(
            "Curated W&B view '%s' is up to date: %s", _VIEW_NAME, workspace.url
        )
    except Exception:
        logger.warning("Could not configure the curated W&B view.", exc_info=True)
