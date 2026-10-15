"""Bias-free linear layer."""

import math

from ..base.backend import Array, uniform
from ..base.module import Module
from ..base.param import Param


class Linear(Module):
    """``y = x @ W`` with no bias (modern blocks drop biases; RMSNorm has no shift either).

    ``W`` has shape (d_in, d_out) and Glorot-uniform init. ``x`` may have any
    number of leading dimensions: (..., d_in) -> (..., d_out).
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()

        limit = math.sqrt(6.0 / (d_in + d_out))
        self.weight = Param(uniform(-limit, limit, (d_in, d_out)))
        self.register_params(self.weight)
        self._x: Array | None = None

    def forward(self, x: Array) -> Array:
        self._x = x
        y = x @ self.weight.value

        return y

    def backward(self, dy: Array) -> Array:
        """dW += x^T dy (summed over every leading dim); returns dx = dy W^T."""
        x = self._x
        if x is None:
            raise RuntimeError("backward() called before forward()")

        d_in = x.shape[-1]
        d_out = dy.shape[-1]
        self.weight.grad += x.reshape(-1, d_in).T @ dy.reshape(-1, d_out)

        return dy @ self.weight.value.T
