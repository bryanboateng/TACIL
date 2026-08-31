import numpy as np
from wandb import Run

from scripts.data.augmentation.stitching.junction_search import (
    JunctionDiagnostics,
    Junctions,
    build_episode_index_per_step,
)
from scripts.shared.distribution_panel import create_distribution_panel
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episode_store import EpisodeStore

STAGE_NAME = "2_data"


def log_junction_statistics(
    *,
    junctions: Junctions,
    junction_diagnostics: JunctionDiagnostics,
    median_expert_step_distance: float,
    junction_radius: float,
    junction_error_limit: float,
    run: Run,
) -> None:
    run.summary[f"{STAGE_NAME}/1_junctions/median_step_distance"] = (
        median_expert_step_distance
    )
    run.summary[f"{STAGE_NAME}/1_junctions/radius"] = junction_radius
    run.summary[f"{STAGE_NAME}/1_junctions/error_limit"] = junction_error_limit
    run.summary[f"{STAGE_NAME}/1_junctions/searchable_source_count"] = (
        junction_diagnostics.searchable_source_count
    )
    run.summary[f"{STAGE_NAME}/1_junctions/candidate_count"] = (
        junction_diagnostics.candidate_count
    )
    run.summary[f"{STAGE_NAME}/1_junctions/accepted_count"] = (
        junction_diagnostics.accepted_count
    )

    # Of the pairs close enough to test, how many the model actually endorsed.
    # A share near one means the radius, not the error, is what limits the
    # search.
    if junction_diagnostics.candidate_count > 0:
        run.summary[f"{STAGE_NAME}/1_junctions/accepted_share_of_candidates"] = (
            junction_diagnostics.accepted_count / junction_diagnostics.candidate_count
        )

    if junctions.errors.size > 0:
        limit_shares = junctions.errors / junction_error_limit
        for label, value in summarize_distribution(values=limit_shares).items():
            run.summary[f"{STAGE_NAME}/2_limit_tightness/{label}"] = value
        run.log(
            data={
                f"{STAGE_NAME}/2_limit_tightness/plot": create_distribution_panel(
                    values=limit_shares, column_name="junction error / limit"
                )
            }
        )


def log_dataset_statistics(
    *,
    episode_store: EpisodeStore,
    stitched_episodes: EpisodeStore,
    kept_junctions: Junctions,
    found_junction_count: int,
    run: Run,
) -> None:
    episode_index_per_step = build_episode_index_per_step(
        episode_end_indices=episode_store.episode_end_indices
    )

    # The whole point is to cross episode boundaries, so this counts how much
    # of the recorded set the kept junctions actually reach. Only the kept ones
    # are counted: the rest never became data.
    source_episode_indices = episode_index_per_step[kept_junctions.source_indices]
    target_episode_indices = episode_index_per_step[kept_junctions.target_indices]
    run.summary[f"{STAGE_NAME}/3_reach/source_episode_count"] = len(
        np.unique(ar=source_episode_indices)
    )
    run.summary[f"{STAGE_NAME}/3_reach/target_episode_count"] = len(
        np.unique(ar=target_episode_indices)
    )
    run.summary[f"{STAGE_NAME}/3_reach/episode_pair_count"] = int(
        np.unique(
            ar=np.stack(
                arrays=[source_episode_indices, target_episode_indices], axis=1
            ),
            axis=0,
        ).shape[0]
    )
    run.summary[f"{STAGE_NAME}/3_reach/found_junction_count"] = found_junction_count
    run.summary[f"{STAGE_NAME}/3_reach/kept_junction_count"] = len(
        kept_junctions.source_indices
    )
    # Below one means the transition budget, not the error limit, is what
    # decides how much stitched data the run gets.
    run.summary[f"{STAGE_NAME}/3_reach/kept_share_of_found"] = (
        len(kept_junctions.source_indices) / found_junction_count
    )

    episode_lengths = np.diff(a=stitched_episodes.episode_end_indices, prepend=0)
    for label, value in summarize_distribution(values=episode_lengths).items():
        run.summary[f"{STAGE_NAME}/4_stitched_length/{label}"] = value

    expert_transition_count = int(episode_store.states.shape[0])
    stitched_transition_count = int(stitched_episodes.states.shape[0])
    total_count = expert_transition_count + stitched_transition_count

    run.summary[f"{STAGE_NAME}/5_dataset/1_recorded_expert_transitions"] = (
        expert_transition_count
    )
    # Every stitched transition but the junction itself is recorded data seen
    # in a new order, so "invented" is generous here. It is the count that
    # matches the other strategies.
    run.summary[f"{STAGE_NAME}/5_dataset/2_invented_transitions"] = (
        stitched_transition_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/3_junction_transitions"] = len(
        kept_junctions.source_indices
    )
    run.summary[f"{STAGE_NAME}/5_dataset/4_total_transitions"] = total_count
    run.summary[f"{STAGE_NAME}/5_dataset/5_invented_fraction"] = (
        stitched_transition_count / total_count
    )
    run.summary[f"{STAGE_NAME}/5_dataset/6_episode_count"] = len(
        episode_store.episode_end_indices
    ) + len(stitched_episodes.episode_end_indices)

    run.log(
        data={
            f"{STAGE_NAME}/4_stitched_length/plot": create_distribution_panel(
                values=episode_lengths, column_name="stitched episode length"
            )
        }
    )
