"""Curated W&B saved view for the TACIL dynamics ensemble.

``ensure_dashboard_view`` is called once per run from the training entrypoint. It upserts
a single saved view named "Ensemble dashboard" whose panels use W&B's native metric
grouping: each panel matches every member's raw metric for one quantity (via a
``<quantity>/member_`` regex) and groups them into a mean line with a filled
standard-deviation band. W&B computes the band; the training code logs only raw
per-member values. It is idempotent: it always targets the same view via a fixed internal
name, so repeated runs update it in place instead of creating duplicates, and no one has
to build panels by hand.

The distribution histograms (``lipschitz/distribution_*``) are still logged and visible in
the project's default view; the workspace API has no histogram panel type, so they cannot
be placed in this curated view.
"""

from typing import Literal

import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws

from scripts.shared.log import create_logger

logger = create_logger()

_VIEW_NAME = "Ensemble dashboard"
# A fixed internal view name makes save() upsert the same saved view every run instead of
# creating a new one; it must keep the "nw-<slug>-v" shape the package round-trips.
_INTERNAL_VIEW_NAME = "nw-tacil-ensemble-dashboard-v"


def _grouped_band_panel(
    *,
    title: str,
    member_metric_regex: str,
    range_function: Literal["minmax", "stddev", "stderr", "none", "samples"] = "stddev",
) -> wr.LinePlot:
    """A panel that groups every member's metric into a mean line with a filled band.

    ``member_metric_regex`` matches the per-member keys for one quantity (e.g.
    ``loss/validation/member_``); W&B aggregates them with mean and the given range
    (``stddev`` for a disagreement band, ``minmax`` to surface the worst member).
    """
    return wr.LinePlot(
        title=title,
        metric_regex=member_metric_regex,
        aggregate=True,
        groupby_aggfunc="mean",
        groupby_rangefunc=range_function,
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
                    _grouped_band_panel(
                        title="training loss",
                        member_metric_regex="loss/training/member_",
                    ),
                    _grouped_band_panel(
                        title="validation loss",
                        member_metric_regex="loss/validation/member_",
                    ),
                ],
            ),
            ws.Section(
                name="Lipschitz",
                is_open=True,
                panels=[
                    _grouped_band_panel(
                        title="mean (training)",
                        member_metric_regex="lipschitz/mean_training/member_",
                    ),
                    _grouped_band_panel(
                        title="mean (validation)",
                        member_metric_regex="lipschitz/mean_validation/member_",
                    ),
                    _grouped_band_panel(
                        title="p99 (training)",
                        member_metric_regex="lipschitz/p99_training/member_",
                    ),
                    _grouped_band_panel(
                        title="p99 (validation)",
                        member_metric_regex="lipschitz/p99_validation/member_",
                    ),
                ],
            ),
            ws.Section(
                name="Health",
                is_open=True,
                panels=[
                    _grouped_band_panel(
                        title="gradient norm",
                        member_metric_regex="gradient_norm/member_",
                        range_function="minmax",
                    )
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
