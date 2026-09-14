"""Tells whether an imagined state ends the episode.

The dynamics model predicts where the robot lands but not whether it fell
over. These rules decide that, and they are the ones OfflineRL-Kit uses in
offlinerlkit/utils/termination_fns.py.

They work on the device the rollout runs on, so that an imagined step never
has to travel to the processor and back.
"""

import torch

from scripts.policies.mopo.dynamics.dynamics import TerminationFunction


def _terminate_halfcheetah(
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    # The half cheetah cannot fall over, so only a state that ran away counts.
    return ~((state_batch > -100.0).all(dim=-1) & (state_batch < 100.0).all(dim=-1))


def _terminate_hopper(
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    height_batch = state_batch[:, 0]
    angle_batch = state_batch[:, 1]
    # The reference writes np.abs(next_obs[:, 1:] < 100), which takes the
    # absolute value of the comparison rather than of the value, and so leaves
    # the comparison one sided. Kept as it stands.
    return ~(
        state_batch.isfinite().all(dim=-1)
        & (state_batch[:, 1:] < 100.0).all(dim=-1)
        & (height_batch > 0.7)
        & (angle_batch.abs() < 0.2)
    )


def _terminate_walker2d(
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    height_batch = state_batch[:, 0]
    angle_batch = state_batch[:, 1]
    return ~(
        (state_batch > -100.0).all(dim=-1)
        & (state_batch < 100.0).all(dim=-1)
        & (height_batch > 0.8)
        & (height_batch < 2.0)
        & (angle_batch > -1.0)
        & (angle_batch < 1.0)
    )


_TERMINATION_FUNCTIONS: dict[str, TerminationFunction] = {
    "halfcheetah": _terminate_halfcheetah,
    "hopper": _terminate_hopper,
    "walker2d": _terminate_walker2d,
}


def get_termination_function(*, robot: str) -> TerminationFunction:
    try:
        return _TERMINATION_FUNCTIONS[robot]
    except KeyError as error:
        raise ValueError(
            f"No termination rule is ported for {robot!r}. Ported are "
            f"{sorted(_TERMINATION_FUNCTIONS)}."
        ) from error
