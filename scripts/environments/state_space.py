from typing import Protocol

import torch

from scripts.environments import angle_arithmetic
from scripts.environments.state_picture import StatePicture


class NetworkInputBuilder(Protocol):
    def __call__(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        ...


class StateSpace:
    def __init__(
        self,
        *,
        dimensionality: int,
        action_dimensionality: int,
        network_input_dimensionality: int,
        angle_indices: tuple[int, ...],
        picture: StatePicture,
        build_network_input: NetworkInputBuilder,
    ) -> None:
        self.dimensionality = dimensionality
        self.action_dimensionality = action_dimensionality
        self.network_input_dimensionality = network_input_dimensionality
        self.picture = picture

        self._angle_indices = angle_indices
        self._build_network_input = build_network_input
        self._angle_mask_per_device: dict[torch.device, torch.Tensor] = {}

    def add(
        self,
        *,
        first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return angle_arithmetic.add(
            first_addend_batch=first_addend_batch,
            second_addend_batch=second_addend_batch,
            angle_mask=self._angle_mask(like=first_addend_batch),
        )

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return angle_arithmetic.subtract(
            minuend_batch=minuend_batch,
            subtrahend_batch=subtrahend_batch,
            angle_mask=self._angle_mask(like=minuend_batch),
        )

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return angle_arithmetic.average(
            state_batch=state_batch, angle_mask=self._angle_mask(like=state_batch)
        )

    def create_network_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        return self._build_network_input(
            state_batch=state_batch, action_batch=action_batch
        )

    # ---- Private --------------------------------------------

    def _angle_mask(
        self, *, like: torch.Tensor
    ) -> torch.Tensor:  # (state_dimensionality,)
        mask = self._angle_mask_per_device.get(like.device)
        if mask is None:
            mask = angle_arithmetic.create_mask(
                dimensionality=self.dimensionality,
                angle_indices=self._angle_indices,
                like=like,
            )
            self._angle_mask_per_device[like.device] = mask
        return mask
