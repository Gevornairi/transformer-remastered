"""One pre-norm encoder block."""

from ..base.backend import Array
from ..base.module import Module
from .dropout import Dropout
from .multi_head_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .rope import RoPE
from .swiglu_feed_forward import SwiGLUFeedForward


class EncoderLayer(Module):
    """Pre-norm block: self-attention then feed-forward, each in a residual branch.

        h = x + drop1(self_attn(norm1(x)))
        y = h + drop2(ffn(norm2(h)))

    Shapes: (batch, T_src, d_model) -> (batch, T_src, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rope: RoPE | None = None,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rope, n_kv_heads)
        self.drop1 = Dropout(dropout)
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.drop2 = Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.register_modules(
            self.self_attn, self.drop1, self.norm1, self.ffn, self.drop2, self.norm2
        )

    def forward(self, x: Array, mask: Array | None = None) -> Array:
        """``mask`` hides PAD source tokens, (batch, 1, 1, T_src)."""
        normed = self.norm1.forward(x)
        x = x + self.drop1.forward(self.self_attn.forward(normed, normed, mask))
        normed = self.norm2.forward(x)
        x = x + self.drop2.forward(self.ffn.forward(normed))

        return x

    def backward(self, dy: Array) -> Array:
        """Each residual passes the gradient through unchanged and adds its branch's."""
        d = dy + self.norm2.backward(self.ffn.backward(self.drop2.backward(dy)))
        d_q, d_kv = self.self_attn.backward(self.drop1.backward(d))

        return d + self.norm1.backward(d_q + d_kv)
