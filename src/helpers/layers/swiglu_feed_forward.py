"""SwiGLU position-wise feed-forward network."""

from ..base.backend import Array
from ..base.module import Module
from .dropout import Dropout
from .linear import Linear
from .silu import SiLU


class SwiGLUFeedForward(Module):
    """``y = W_down( dropout( silu(x W_gate) * (x W_up) ) )``, applied to each position.

    Three matrices instead of the ReLU FFN's two, so d_ff ~ 8/3 * d_model
    (1344 for d_model = 512) keeps the parameter count of the original
    d_ff = 4 * d_model. Shapes: (..., d_model) -> (..., d_model).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self._cache: tuple[Array, Array] | None = None  # (gate, up)
        self.w_gate = Linear(d_model, d_ff)
        self.w_up = Linear(d_model, d_ff)
        self.w_down = Linear(d_ff, d_model)
        self.silu = SiLU()
        self.dropout = Dropout(dropout)
        self.register_modules(
            self.w_gate, self.w_up, self.w_down, self.silu, self.dropout
        )

    def forward(self, x: Array) -> Array:
        gate = self.silu.forward(self.w_gate.forward(x))
        up = self.w_up.forward(x)
        self._cache = (gate, up)

        return self.w_down.forward(self.dropout.forward(gate * up))

    def backward(self, dy: Array) -> Array:
        """Through the product rule on gate * up; x feeds both branches, so their dx add."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        gate, up = self._cache

        d_hidden = self.dropout.backward(self.w_down.backward(dy))
        d_gate = d_hidden * up
        d_up = d_hidden * gate
        d_gate_pre = self.silu.backward(d_gate)

        return self.w_gate.backward(d_gate_pre) + self.w_up.backward(d_up)
