from typing import Self

import torch

from scripts.shared.normalization import ZScoreNormalizer


class Normalizers(torch.nn.Module):
    state_normalizer: ZScoreNormalizer
    network_input_normalizer: ZScoreNormalizer
    state_displacement_normalizer: ZScoreNormalizer

    def __init__(
        self,
        *,
        state_normalizer: ZScoreNormalizer,
        network_input_normalizer: ZScoreNormalizer,
        state_displacement_normalizer: ZScoreNormalizer,
    ):
        super().__init__()
        self.state_normalizer = state_normalizer
        self.network_input_normalizer = network_input_normalizer
        self.state_displacement_normalizer = state_displacement_normalizer

    @classmethod
    def from_transitions(
        cls,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        raw_network_input_batch: torch.Tensor,  # (batch_size, NETWORK_INPUT_DIMENSIONALITY)
        state_displacement_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> Self:
        return cls(
            state_normalizer=ZScoreNormalizer.from_values(value_batch=state_batch),
            network_input_normalizer=ZScoreNormalizer.from_values(
                value_batch=raw_network_input_batch
            ),
            state_displacement_normalizer=ZScoreNormalizer.from_values(
                value_batch=state_displacement_batch
            ),
        )
