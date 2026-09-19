import torch


class StateSpace:
    def __init__(self, *, dimensionality: int, angle_indices: tuple[int, ...]) -> None:
        self.dimensionality = dimensionality

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
            condition=self._get_angle_mask(like=sum_batch),
            input=sum_batch % _FULL_TURN,
            other=sum_batch,
        )

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        difference_batch = minuend_batch - subtrahend_batch
        return torch.where(
            condition=self._get_angle_mask(like=difference_batch),
            input=(difference_batch + torch.pi) % _FULL_TURN - torch.pi,
            other=difference_batch,
        )

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        average_batch = state_batch.mean(dim=0)
        return torch.where(
            condition=self._get_angle_mask(like=average_batch),
            input=torch.atan2(
                input=torch.sin(input=state_batch).mean(dim=0),
                other=torch.cos(input=state_batch).mean(dim=0),
            )
            % _FULL_TURN,
            other=average_batch,
        )

    # ---- Private --------------------------------------------

    def _get_angle_mask(
        self, *, like: torch.Tensor
    ) -> torch.Tensor:  # (state_dimensionality,)
        if like.device not in self._angle_mask_per_device:
            self._angle_mask_per_device[like.device] = _create_angle_mask(
                dimensionality=self.dimensionality,
                angle_indices=self._angle_indices,
                device=like.device,
            )
        return self._angle_mask_per_device[like.device]


# ---- Private --------------------------------------------


_FULL_TURN = 2.0 * torch.pi


def _create_angle_mask(
    *, dimensionality: int, angle_indices: tuple[int, ...], device: torch.device
) -> torch.Tensor:  # (state_dimensionality,)
    mask = torch.zeros(dimensionality, dtype=torch.bool, device=device)
    mask[list(angle_indices)] = True
    return mask
