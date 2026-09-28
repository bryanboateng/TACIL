import torch

from scripts.data.augmentation.tacil.create_dynamics.data import Transitions
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder
from scripts.shared.normalization import ZScoreNormalizer


class Normalizers(torch.nn.Module):
    def __init__(
        self,
        *,
        state: ZScoreNormalizer,
        network_input: ZScoreNormalizer,
        state_displacement: ZScoreNormalizer,
    ) -> None:
        super().__init__()
        self.state = state
        self.network_input = network_input
        self.state_displacement = state_displacement


def create_normalizers(
    *,
    transitions: Transitions,
    state_space: StateSpace,
    network_input_builder: TacilNetworkInputBuilder,
    device: torch.device,
) -> Normalizers:
    state_batch = torch.from_numpy(transitions.states).to(
        device=device
    )  # (transition_count, state_dimensionality)
    next_state_batch = torch.from_numpy(transitions.next_states).to(
        device=device
    )  # (transition_count, state_dimensionality)
    action_batch = torch.from_numpy(transitions.actions).to(
        device=device
    )  # (transition_count, action_dimensionality)
    return Normalizers(
        state=ZScoreNormalizer(value_batch=state_batch),
        network_input=ZScoreNormalizer(
            value_batch=network_input_builder(
                state_batch=state_batch, action_batch=action_batch
            )
        ),
        state_displacement=ZScoreNormalizer(
            value_batch=state_space.subtract(
                minuend_batch=next_state_batch, subtrahend_batch=state_batch
            )
        ),
    ).to(device=device)
