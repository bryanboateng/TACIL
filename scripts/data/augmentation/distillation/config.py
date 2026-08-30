from dataclasses import dataclass

from scripts.shared.config_checks import check_positive


@dataclass(frozen=True)
class DistillationDataConfig:
    # GPI's own parameters, as in its policy: how far to follow the recorded
    # action, how hard to pull toward the recorded position, and how many
    # recorded steps to walk before looking for a new one. The default plan
    # length is GPI's own default.
    gpi_progression_weight: float = 1.0
    gpi_attraction_weight: float = 1.0
    gpi_plan_length: int = 8

    # Std of the block displacement applied to each anchor, as a fraction of
    # each block dimension's range in the expert data. Only the block moves:
    # the evaluation disturbances hit the block, and a displaced agent needs no
    # teacher to find its way back.
    disturbance_range_fraction: float = 0.15

    # How many steps GPI is followed through the dynamics model. Every step
    # adds model error, so this trades coverage of GPI's behaviour against how
    # real the states it is demonstrated on still are.
    max_chain_length: int = 16

    # A chain is back on the demonstrations, and stops, when it comes within
    # this many median expert steps of a recorded state. It is also the filter
    # on the start: a disturbance that already leaves the state this close to
    # the demonstrations has nothing to teach.
    on_manifold_tolerance_step_multiple: float = 1.0

    def __post_init__(self) -> None:
        check_positive(
            # Both weights scale a pull; at zero GPI would stand still, and
            # below zero it would move away from what it is following.
            gpi_progression_weight=self.gpi_progression_weight,
            gpi_attraction_weight=self.gpi_attraction_weight,
            gpi_plan_length=self.gpi_plan_length,
            # Without a disturbance every state starts on the demonstrations,
            # so no chain would be followed and nothing would be learned.
            disturbance_range_fraction=self.disturbance_range_fraction,
            max_chain_length=self.max_chain_length,
            on_manifold_tolerance_step_multiple=(
                self.on_manifold_tolerance_step_multiple
            ),
        )
