import torch

from scripts.environments import angle_arithmetic
from scripts.environments.network_input import NetworkInput
from scripts.environments.plot_layout import PlotLayout


class StateSpace:
    def __init__(
        self,
        *,
        dimensionality: int,
        angle_indices: tuple[int, ...],
        network_input: NetworkInput,
        plot_layout: PlotLayout,
    ) -> None:
        self.dimensionality = dimensionality
        self.network_input = network_input
        self.plot_layout = plot_layout

        self._angle_indices = angle_indices
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
