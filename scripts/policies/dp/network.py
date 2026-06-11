import math

import torch


class ConditionalUnet1d(torch.nn.Module):
    def __init__(
        self,
        input_channel_count: int,
        condition_dimensionality: int,
        hidden_channel_counts: tuple[int, ...],
    ) -> None:
        super().__init__()

        kernel_size = 5
        channel_counts = [input_channel_count, *hidden_channel_counts]
        first_hidden_channel_count = hidden_channel_counts[0]

        diffusion_timestep_embedding_dimensionality = 256
        self._diffusion_timestep_encoder = torch.nn.Sequential(
            SinusoidalPositionalEmbedding(
                dimensionality=diffusion_timestep_embedding_dimensionality
            ),
            torch.nn.Linear(
                in_features=diffusion_timestep_embedding_dimensionality,
                out_features=diffusion_timestep_embedding_dimensionality * 4,
            ),
            torch.nn.Mish(),
            torch.nn.Linear(
                in_features=diffusion_timestep_embedding_dimensionality * 4,
                out_features=diffusion_timestep_embedding_dimensionality,
            ),
        )
        combined_condition_dimensionality = (
            diffusion_timestep_embedding_dimensionality + condition_dimensionality
        )

        level_channel_count_pairs = list(zip(channel_counts[:-1], channel_counts[1:]))
        bottleneck_channel_count = channel_counts[-1]
        last_level_index = len(level_channel_count_pairs) - 1

        self._middle_blocks = torch.nn.ModuleList(
            modules=[
                ConditionalResidualBlock1d(
                    input_channel_count=bottleneck_channel_count,
                    output_channel_count=bottleneck_channel_count,
                    condition_dimensionality=combined_condition_dimensionality,
                    kernel_size=kernel_size,
                ),
                ConditionalResidualBlock1d(
                    input_channel_count=bottleneck_channel_count,
                    output_channel_count=bottleneck_channel_count,
                    condition_dimensionality=combined_condition_dimensionality,
                    kernel_size=kernel_size,
                ),
            ]
        )

        self._downsample_levels = torch.nn.ModuleList(
            modules=[
                _DownsampleLevel(
                    input_channel_count=level_input_channel_count,
                    output_channel_count=level_output_channel_count,
                    condition_dimensionality=combined_condition_dimensionality,
                    kernel_size=kernel_size,
                    apply_downsample=level_index < last_level_index,
                )
                for level_index, (
                    level_input_channel_count,
                    level_output_channel_count,
                ) in enumerate(iterable=level_channel_count_pairs)
            ]
        )

        self._upsample_levels = torch.nn.ModuleList(
            modules=[
                _UpsampleLevel(
                    input_channel_count=level_input_channel_count,
                    output_channel_count=level_output_channel_count,
                    condition_dimensionality=combined_condition_dimensionality,
                    kernel_size=kernel_size,
                    apply_upsample=level_index < last_level_index,
                )
                for level_index, (
                    level_output_channel_count,
                    level_input_channel_count,
                ) in enumerate(iterable=reversed(level_channel_count_pairs[1:]))
            ]
        )

        self._final_convolution = torch.nn.Sequential(
            Conv1dBlock(
                input_channel_count=first_hidden_channel_count,
                output_channel_count=first_hidden_channel_count,
                kernel_size=kernel_size,
            ),
            torch.nn.Conv1d(
                in_channels=first_hidden_channel_count,
                out_channels=input_channel_count,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        noisy_sample_batch: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        condition_batch: torch.Tensor,
    ) -> torch.Tensor:
        noisy_sample_batch = noisy_sample_batch.moveaxis(source=-1, destination=-2)

        diffusion_timestep_batch = _broadcast_to_batch(
            values=diffusion_timestep, reference_batch=noisy_sample_batch
        )
        combined_condition_batch = torch.cat(
            tensors=[
                self._diffusion_timestep_encoder(input=diffusion_timestep_batch),
                condition_batch,
            ],
            dim=-1,
        )

        hidden_state_batch = noisy_sample_batch
        skip_connection_batches: list[torch.Tensor] = []
        for downsample_level in self._downsample_levels:
            hidden_state_batch, skip_connection_batch = downsample_level(
                input_batch=hidden_state_batch, condition_batch=combined_condition_batch
            )
            skip_connection_batches.append(skip_connection_batch)

        for middle_block in self._middle_blocks:
            hidden_state_batch = middle_block(
                input_batch=hidden_state_batch, condition_batch=combined_condition_batch
            )

        for upsample_level in self._upsample_levels:
            hidden_state_batch = upsample_level(
                input_batch=hidden_state_batch,
                skip_connection_batch=skip_connection_batches.pop(),
                condition_batch=combined_condition_batch,
            )

        hidden_state_batch = self._final_convolution(input=hidden_state_batch)
        return hidden_state_batch.moveaxis(source=-1, destination=-2)


class _DownsampleLevel(torch.nn.Module):
    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        condition_dimensionality: int,
        kernel_size: int,
        apply_downsample: bool,
    ) -> None:
        super().__init__()
        self._first_residual_block = ConditionalResidualBlock1d(
            input_channel_count=input_channel_count,
            output_channel_count=output_channel_count,
            condition_dimensionality=condition_dimensionality,
            kernel_size=kernel_size,
        )
        self._second_residual_block = ConditionalResidualBlock1d(
            input_channel_count=output_channel_count,
            output_channel_count=output_channel_count,
            condition_dimensionality=condition_dimensionality,
            kernel_size=kernel_size,
        )
        self._downsample = (
            Downsample1d(channel_count=output_channel_count)
            if apply_downsample
            else torch.nn.Identity()
        )

    def forward(
        self, input_batch: torch.Tensor, condition_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state_batch = self._first_residual_block(
            input_batch=input_batch, condition_batch=condition_batch
        )
        hidden_state_batch = self._second_residual_block(
            input_batch=hidden_state_batch, condition_batch=condition_batch
        )
        skip_connection_batch = hidden_state_batch

        hidden_state_batch = self._downsample(hidden_state_batch)
        return hidden_state_batch, skip_connection_batch


class _UpsampleLevel(torch.nn.Module):
    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        condition_dimensionality: int,
        kernel_size: int,
        apply_upsample: bool,
    ) -> None:
        super().__init__()
        # The first residual block sees input_channel_count * 2 because the skip
        # connection (same channel count as the incoming hidden state) is
        # concatenated along the channel dimension before this block runs.
        self._first_residual_block = ConditionalResidualBlock1d(
            input_channel_count=input_channel_count * 2,
            output_channel_count=output_channel_count,
            condition_dimensionality=condition_dimensionality,
            kernel_size=kernel_size,
        )
        self._second_residual_block = ConditionalResidualBlock1d(
            input_channel_count=output_channel_count,
            output_channel_count=output_channel_count,
            condition_dimensionality=condition_dimensionality,
            kernel_size=kernel_size,
        )
        self._upsample = (
            Upsample1d(channel_count=output_channel_count)
            if apply_upsample
            else torch.nn.Identity()
        )

    def forward(
        self,
        input_batch: torch.Tensor,
        skip_connection_batch: torch.Tensor,
        condition_batch: torch.Tensor,
    ) -> torch.Tensor:
        hidden_state_batch = torch.cat(
            tensors=(input_batch, skip_connection_batch), dim=1
        )
        hidden_state_batch = self._first_residual_block(
            input_batch=hidden_state_batch, condition_batch=condition_batch
        )
        hidden_state_batch = self._second_residual_block(
            input_batch=hidden_state_batch, condition_batch=condition_batch
        )
        return self._upsample(hidden_state_batch)


def _broadcast_to_batch(
    values: torch.Tensor, reference_batch: torch.Tensor
) -> torch.Tensor:
    if values.ndim == 0:
        values = values[None]
    return values.to(device=reference_batch.device).expand(reference_batch.shape[0])


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dimensionality: int) -> None:
        super().__init__()
        self._dimensionality = dimensionality

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        half_dimensionality = self._dimensionality // 2
        log_frequency_spacing = math.log(10000) / (half_dimensionality - 1)
        frequencies = torch.exp(
            input=torch.arange(end=half_dimensionality, device=input_batch.device)
            * -log_frequency_spacing
        )
        phase_angle_batch = input_batch[:, None] * frequencies[None, :]
        return torch.cat(
            tensors=(phase_angle_batch.sin(), phase_angle_batch.cos()), dim=-1
        )


class ConditionalResidualBlock1d(torch.nn.Module):
    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        condition_dimensionality: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self._output_channel_count = output_channel_count

        self._first_convolution_block = Conv1dBlock(
            input_channel_count=input_channel_count,
            output_channel_count=output_channel_count,
            kernel_size=kernel_size,
        )
        self._second_convolution_block = Conv1dBlock(
            input_channel_count=output_channel_count,
            output_channel_count=output_channel_count,
            kernel_size=kernel_size,
        )

        # FiLM (https://arxiv.org/abs/1709.07871): per-channel scale and bias.
        self._feature_modulation_encoder = torch.nn.Sequential(
            torch.nn.Mish(),
            torch.nn.Linear(
                in_features=condition_dimensionality,
                out_features=output_channel_count * 2,
            ),
            torch.nn.Unflatten(dim=-1, unflattened_size=(-1, 1)),
        )
        self._residual_projection = (
            torch.nn.Conv1d(
                in_channels=input_channel_count,
                out_channels=output_channel_count,
                kernel_size=1,
            )
            if input_channel_count != output_channel_count
            else torch.nn.Identity()
        )

    def forward(
        self, input_batch: torch.Tensor, condition_batch: torch.Tensor
    ) -> torch.Tensor:
        hidden_state_batch = self._first_convolution_block(input_batch=input_batch)
        modulation_parameter_batch = self._feature_modulation_encoder(
            input=condition_batch
        ).reshape(-1, 2, self._output_channel_count, 1)
        scale_batch = modulation_parameter_batch[:, 0]
        shift_batch = modulation_parameter_batch[:, 1]
        hidden_state_batch = scale_batch * hidden_state_batch + shift_batch
        hidden_state_batch = self._second_convolution_block(
            input_batch=hidden_state_batch
        )
        return hidden_state_batch + self._residual_projection(input=input_batch)


class Downsample1d(torch.nn.Module):
    def __init__(self, channel_count: int) -> None:
        super().__init__()
        self._convolution = torch.nn.Conv1d(
            in_channels=channel_count,
            out_channels=channel_count,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self._convolution(input=input_batch)


class Upsample1d(torch.nn.Module):
    def __init__(self, channel_count: int) -> None:
        super().__init__()
        self._transposed_convolution = torch.nn.ConvTranspose1d(
            in_channels=channel_count,
            out_channels=channel_count,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self._transposed_convolution(input=input_batch)


class Conv1dBlock(torch.nn.Module):
    def __init__(
        self, input_channel_count: int, output_channel_count: int, kernel_size: int
    ) -> None:
        super().__init__()
        self._layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=input_channel_count,
                out_channels=output_channel_count,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            torch.nn.GroupNorm(num_groups=8, num_channels=output_channel_count),
            torch.nn.Mish(),
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self._layers(input=input_batch)
