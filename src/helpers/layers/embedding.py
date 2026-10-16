"""Token embedding table."""

import math

from ..base.backend import Array, normal, scatter_add
from ..base.module import Module
from ..base.param import Param


class Embedding(Module):
    """Look up a row of ``weight`` per token id and scale it by sqrt(d_model).

    ``weight`` (vocab, d_model) is initialised with std d_model^-0.5, so the
    scaled embeddings start with roughly unit RMS. With shared embeddings the
    same table is also the output projection (``Transformer.project``).
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = Param(normal(d_model**-0.5, (vocab_size, d_model)))
        self.register_params(self.weight)
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

    def forward(self, ids: Array) -> Array:
        """``ids`` (batch, T) int -> (batch, T, d_model)."""
        return self.weight.value[ids] * self.scale

    def backward(self, dy: Array, ids: Array) -> None:
        """Add ``dy * scale`` into the gradient rows of ``ids``; tokens not in the batch get none.

        ``ids`` is passed in rather than cached: a shared table runs forward
        for both encoder and decoder before either runs backward, so a cache
        here would only hold the decoder's ids.
        """
        self.weight.grad = scatter_add(self.weight.grad, ids, dy * self.scale)
