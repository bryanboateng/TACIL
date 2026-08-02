import torch


class LipschitzConstrainedLinear(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimensionality: int,
        output_dimensionality: int,
        lipschitz_constant_upper_bound: float,
    ):
        super().__init__()
        self._linear = torch.nn.Linear(
            in_features=input_dimensionality, out_features=output_dimensionality
        )
        torch.nn.utils.parametrize.register_parametrization(
            module=self._linear,
            tensor_name="weight",
            parametrization=_SpectralNormBound(
                weight=self._linear.weight,
                lipschitz_constant_upper_bound=lipschitz_constant_upper_bound,
            ),
        )

    def forward(
        self,
        input_batch: torch.Tensor,  # (batch_size, input_dimensionality)
    ) -> torch.Tensor:  # (batch_size, output_dimensionality)
        return self._linear(input_batch)


# ---- Private --------------------------------------------


class _SpectralNormBound(torch.nn.Module):
    """Caps a weight matrix's spectral norm at `lipschitz_constant_upper_bound`.

    Follows Eq. 3 of CCIL (arXiv:2405.19307), the paper this work builds on:

        W <- W / max(sigma(W) / c, 1)

    The `max` is what makes this a ceiling rather than a fixed value. A weight
    whose spectral norm is already below `c` passes through untouched, so a
    layer is free to be gentler than the bound — the paper relies on the local
    Lipschitz constants varying below it rather than sitting on it.

    `sigma` is estimated by power iteration rather than an exact SVD, following
    Miyato et al. (arXiv:1802.05957). One iteration per training forward pass
    suffices because the vectors persist across steps while the weights move
    only slightly between them, and it costs two matrix-vector products instead
    of a decomposition. It also differentiates cleanly: with the vectors held
    fixed, the derivative of their bilinear form is just their outer product.
    """

    # Estimates of the leading left and right singular vectors.
    left_vector: torch.Tensor  # (output_dimensionality,)
    right_vector: torch.Tensor  # (input_dimensionality,)

    def __init__(self, *, weight: torch.Tensor, lipschitz_constant_upper_bound: float):
        super().__init__()
        output_dimensionality, input_dimensionality = weight.shape
        self._lipschitz_constant_upper_bound = lipschitz_constant_upper_bound
        self.register_buffer(
            name="left_vector",
            tensor=torch.nn.functional.normalize(
                input=torch.randn(output_dimensionality), dim=0
            ),
        )
        self.register_buffer(
            name="right_vector",
            tensor=torch.nn.functional.normalize(
                input=torch.randn(input_dimensionality), dim=0
            ),
        )

    def forward(
        self,
        weight: torch.Tensor,  # (output_dimensionality, input_dimensionality)
    ) -> torch.Tensor:  # (output_dimensionality, input_dimensionality)
        left_vector, right_vector = self.left_vector, self.right_vector

        if self.training:
            # Refine the estimate in place, so the next step picks up where this
            # one left off. Kept out of the graph: the vectors estimate a
            # property of the weight, they are not a function of it to
            # differentiate through.
            with torch.no_grad():
                torch.nn.functional.normalize(
                    input=weight.t() @ left_vector, dim=0, out=self.right_vector
                )
                torch.nn.functional.normalize(
                    input=weight @ self.right_vector, dim=0, out=self.left_vector
                )
            # Work on copies from here. A loss that runs this layer more than
            # once before its backward pass — as the multi-step rollout in the
            # training objective does — would otherwise refine the very vectors
            # an earlier graph was built on, and autograd rejects that.
            left_vector = self.left_vector.clone()
            right_vector = self.right_vector.clone()

        spectral_norm = torch.dot(input=left_vector, tensor=weight @ right_vector)
        return weight / torch.clamp(
            input=spectral_norm / self._lipschitz_constant_upper_bound, min=1.0
        )
