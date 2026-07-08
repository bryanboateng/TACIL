import torch

from scripts.data.augmentation.tacil.mmmmmmm.dynamics_network import DynamicsNetwork
from scripts.data.augmentation.tacil.mmmmmmm.normalizers import Normalizers


class DynamicsModelEnsemble(torch.nn.Module):
    def __init__(self, *, normalizers: Normalizers, networks: list[DynamicsNetwork]):
        super().__init__()
        self.normalizers = normalizers
        self._networks = torch.nn.ModuleList(modules=networks)

    @property
    def state_dimensionality(self) -> int:
        return self._first_network.state_dimensionality

    @property
    def action_dimensionality(self) -> int:
        return self._first_network.action_dimensionality

    @property
    def _first_network(self) -> DynamicsNetwork:
        first_network = self._networks[0]
        assert isinstance(first_network, DynamicsNetwork)
        return first_network

    def normalize_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.normalizers.normalize_state(state_batch=state_batch)

    def denormalize_state(
        self,
        *,
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.normalizers.denormalize_state(
            normalized_state_batch=normalized_state_batch
        )

    def predict_next_state(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return self.predict_next_state_per_member(
            state_batch=state_batch, action_batch=action_batch
        ).mean(dim=0)

    def predict_next_state_per_member(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (member_count, batch_size, state_dimensionality)
        return torch.stack(
            tensors=[
                network.predict_next_state(
                    normalizers=self.normalizers,
                    state_batch=state_batch,
                    action_batch=action_batch,
                )
                for network in self.networks
            ],
            dim=0,
        )

    def local_lipschitz_constants(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        return torch.stack(
            tensors=[
                network.local_lipschitz_constants(
                    normalizers=self.normalizers,
                    state_batch=state_batch,
                    action_batch=action_batch,
                )
                for network in self.networks
            ],
            dim=0,
        ).mean(dim=0)

    @property
    def networks(self) -> list[DynamicsNetwork]:
        # ModuleList is typed as nn.Module but only ever holds DynamicsNetworks.
        networks: list[DynamicsNetwork] = []
        for network in self._networks:
            assert isinstance(network, DynamicsNetwork)
            networks.append(network)
        return networks
