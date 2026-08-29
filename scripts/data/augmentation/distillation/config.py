from dataclasses import dataclass


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
