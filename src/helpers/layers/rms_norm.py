"""Root-mean-square layer normalisation."""

from ..base.backend import Array, xp
from ..base.module import Module
from ..base.param import Param


class RMSNorm(Module):
    """``y = gain * x / sqrt(mean(x^2) + eps)`` over the last axis.

    LayerNorm without the mean subtraction and the bias; ``gain`` starts at 1.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gain = Param(xp.ones(d_model))
        self.register_params(self.gain)
        self.eps = eps
        self._cache: tuple[Array, Array] | None = None  # (x_hat, 1/rms)

    def forward(self, x: Array) -> Array:
        inv_rms = 1.0 / xp.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        x_hat = x * inv_rms
        self._cache = (x_hat, inv_rms)

        return self.gain.value * x_hat

    def backward(self, dy: Array) -> Array:
        """dx = (dx_hat - x_hat * mean(dx_hat * x_hat)) / rms, with dx_hat = dy * gain."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        x_hat, inv_rms = self._cache
        dim = x_hat.shape[-1]
        self.gain.grad += (dy * x_hat).reshape(-1, dim).sum(axis=0)
        dx_hat = dy * self.gain.value
        dx = inv_rms * (dx_hat - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True))

        return dx
