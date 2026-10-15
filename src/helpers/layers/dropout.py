"""Inverted dropout."""

from ..base.backend import Array, keep_mask
from ..base.module import Module


class Dropout(Module):
    """Zero each element with probability ``p`` during training.

    Kept elements are scaled by 1/(1-p) so the expected activation is the same
    in training and evaluation; in eval mode (or with p = 0) it is the identity.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
        self.p = p
        self._mask = None  # scaled keep-mask of the last forward, None if identity

    def forward(self, x: Array) -> Array:
        if not self.training or self.p == 0.0:
            self._mask = None

            return x

        self._mask = keep_mask(1.0 - self.p, x.shape) * (1.0 / (1.0 - self.p))

        return x * self._mask

    def backward(self, dy: Array) -> Array:
        """Same mask as the forward pass: dropped elements get no gradient."""
        return dy if self._mask is None else dy * self._mask
