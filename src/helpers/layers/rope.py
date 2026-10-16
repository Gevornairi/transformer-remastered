"""Rotary position embedding (Su et al., 2021)."""

import numpy as np

from ..base.backend import DTYPE, Array, asarray, xp
from ..base.module import Module


class RoPE(Module):
    """Rotate each query/key head vector by an angle proportional to its position.

    Uses the "half" pairing: dimension i is rotated together with i + d_k/2, at
    frequency theta_i = base^(-2i/d_k). The dot product of a query at position
    m and a key at position n then depends only on n - m, which is how the
    model sees relative position. Applied in self-attention only.

    The cos/sin tables are precomputed for positions 0..max_len-1. The module
    has no parameters and no cache, so one instance can serve every layer.
    """

    def __init__(self, d_k: int, max_len: int = 5000, base: float = 10000.0) -> None:
        super().__init__()
        if d_k % 2:
            raise ValueError(f"d_k={d_k} must be even for RoPE")
        half = d_k // 2
        theta = base ** (-np.arange(half, dtype=np.float64) * 2 / d_k)
        angles = np.arange(max_len, dtype=np.float64)[:, None] * theta[None, :]
        angles = np.concatenate([angles, angles], axis=-1)  # (max_len, d_k)
        self.cos = asarray(np.cos(angles), DTYPE)
        self.sin = asarray(np.sin(angles), DTYPE)
        self.half = half
        self.max_len = max_len

    def _rotate_half(self, x: Array) -> Array:
        """[x1, x2] -> [-x2, x1], where x1 and x2 are the two halves of the last axis."""
        x1, x2 = x[..., : self.half], x[..., self.half :]
        return xp.concatenate([-x2, x1], axis=-1)

    def _tables(self, x: Array) -> tuple[Array, Array]:
        """cos/sin rows for positions 0..T-1, where T = x.shape[-2]."""
        t = x.shape[-2]
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds RoPE max_len={self.max_len}")
        return self.cos[:t], self.sin[:t]

    def forward(self, x: Array) -> Array:
        """x (batch, heads, T, d_k) -> the same shape, rotated by position along T."""
        cos, sin = self._tables(x)
        return x * cos + self._rotate_half(x) * sin

    def backward(self, dy: Array) -> Array:
        """Rotate the gradient back by the same angle: dx = dy*cos - rotate_half(dy*sin)."""
        cos, sin = self._tables(dy)
        return dy * cos - self._rotate_half(dy * sin)
