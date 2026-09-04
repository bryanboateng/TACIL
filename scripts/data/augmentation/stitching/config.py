from dataclasses import dataclass


@dataclass(frozen=True)
class StitchingDataConfig:
    # How close two states from different episodes must be before the pair is
    # worth testing, in multiples of the median expert step distance. This only
    # narrows the search; what decides a junction is the error below.
    junction_radius_step_multiple: float = 2.0

    # Ceiling on a junction's error: how far the model says the borrowed action
    # lands from the state it is supposed to reach, again in multiples of the
    # median expert step distance.
    junction_error_limit_step_multiple: float = 1.0

    # How much stitched data to keep, as a multiple of the expert transition
    # count. Almost every state has a plausible partner somewhere, so without a
    # budget the stitched set would dwarf the recorded one and the policy would
    # train mostly on splices. The junctions with the smallest error are kept.
    transition_budget_multiple: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("junction_radius_step_multiple", self.junction_radius_step_multiple),
            (
                "junction_error_limit_step_multiple",
                self.junction_error_limit_step_multiple,
            ),
            ("transition_budget_multiple", self.transition_budget_multiple),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")
        if self.junction_error_limit_step_multiple > self.junction_radius_step_multiple:
            raise ValueError(
                f"The junction error limit "
                f"({self.junction_error_limit_step_multiple}) is wider than the "
                f"search radius ({self.junction_radius_step_multiple}), so the "
                f"radius decides which junctions are found and the limit never "
                f"binds. Widen the radius instead."
            )
