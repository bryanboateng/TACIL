from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt


@dataclass(frozen=True)
class Bound:
    entries: slice
    minimum: float
    maximum: float


class StateSpace:
    def __init__(
        self,
        *,
        dimensionality: int,
        angle_indices: tuple[int, ...],
        bounds: tuple[Bound, ...],
    ) -> None:
        self.dimensionality = dimensionality

        self._bounds = bounds

        self._angle_indices = angle_indices
        self._angle_mask_per_device: dict[torch.device, torch.Tensor] = {}

    def add(
        self,
        *,
        first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        sum_batch = first_addend_batch + second_addend_batch
        return torch.where(
            self._angle_mask(like=first_addend_batch), sum_batch % _FULL_TURN, sum_batch
        )

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        difference_batch = minuend_batch - subtrahend_batch
        return torch.where(
            self._angle_mask(like=minuend_batch),
            (difference_batch + torch.pi) % _FULL_TURN - torch.pi,
            difference_batch,
        )

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return torch.where(
            self._angle_mask(like=state_batch),
            torch.atan2(
                input=torch.sin(input=state_batch).mean(dim=0),
                other=torch.cos(input=state_batch).mean(dim=0),
            )
            % _FULL_TURN,
            state_batch.mean(dim=0),
        )

    def contains(
        self,
        *,
        state_batch: npt.NDArray[np.float32],  # (batch_size, state_dimensionality)
    ) -> npt.NDArray[np.bool_]:  # (batch_size,)
        inside_batch = np.isfinite(state_batch).all(axis=-1)  # (batch_size,)
        for bound in self._bounds:
            entry_batch = state_batch[:, bound.entries]
            inside_batch = (
                inside_batch
                & (entry_batch > bound.minimum).all(axis=-1)
                & (entry_batch < bound.maximum).all(axis=-1)
            )
        return np.asarray(a=inside_batch)

    # ---- Private --------------------------------------------

    def _angle_mask(
        self, *, like: torch.Tensor
    ) -> torch.Tensor:  # (state_dimensionality,)
        mask = self._angle_mask_per_device.get(like.device)
        if mask is None:
            mask = torch.zeros(
                self.dimensionality, dtype=torch.bool, device=like.device
            )
            mask[list(self._angle_indices)] = True
            self._angle_mask_per_device[like.device] = mask
        return mask


# ---- Private --------------------------------------------


_FULL_TURN = 2.0 * torch.pi
