"""One pre-norm decoder block."""

from ..base.backend import Array
from ..base.module import Module
from .dropout import Dropout
from .multi_head_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .rope import RoPE
from .swiglu_feed_forward import SwiGLUFeedForward


class DecoderLayer(Module):
    """Pre-norm block: causal self-attention, cross-attention, feed-forward.

        h1 = x  + drop1(self_attn(norm1(x)))
        h2 = h1 + drop2(cross_attn(norm2(h1), memory))
        y  = h2 + drop3(ffn(norm3(h2)))

    RoPE is used in self-attention only: in cross-attention queries and keys
    come from different sequences, so their relative position means nothing.

    Shapes: x (batch, T_tgt, d_model), memory (batch, T_src, d_model)
    -> (batch, T_tgt, d_model).
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
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, dropout, n_kv_heads=n_kv_heads
        )
        self.drop2 = Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.drop3 = Dropout(dropout)
        self.norm3 = RMSNorm(d_model)
        self.register_modules(
            self.self_attn,
            self.drop1,
            self.norm1,
            self.cross_attn,
            self.drop2,
            self.norm2,
            self.ffn,
            self.drop3,
            self.norm3,
        )

    def forward(
        self,
        x: Array,
        memory: Array,
        src_mask: Array | None = None,
        tgt_mask: Array | None = None,
    ) -> Array:
        """``src_mask`` (batch, 1, 1, T_src) hides PAD memory; ``tgt_mask`` (batch, 1, T_tgt, T_tgt) is causal + padding."""
        normed = self.norm1.forward(x)
        x = x + self.drop1.forward(self.self_attn.forward(normed, normed, tgt_mask))
        normed = self.norm2.forward(x)
        x = x + self.drop2.forward(self.cross_attn.forward(normed, memory, src_mask))
        normed = self.norm3.forward(x)
        x = x + self.drop3.forward(self.ffn.forward(normed))

        return x

    def backward(self, dy: Array) -> tuple[Array, Array]:
        """Returns (dx, d_memory); d_memory is this layer's share of the memory gradient."""
        d = dy + self.norm3.backward(self.ffn.backward(self.drop3.backward(dy)))

        d_q_cross, d_memory = self.cross_attn.backward(self.drop2.backward(d))
        d = d + self.norm2.backward(d_q_cross)

        d_q_self, d_kv_self = self.self_attn.backward(self.drop1.backward(d))
        d = d + self.norm1.backward(d_q_self + d_kv_self)

        return d, d_memory
