"""Multi-head scaled dot-product attention with optional RoPE and grouped-query attention."""

import math

from ..base.backend import Array, xp
from ..base.module import Module
from ..base.softmax import softmax, softmax_backward
from .dropout import Dropout
from .linear import Linear
from .rope import RoPE

NEG_INF = -1e9  # score given to masked positions; exp() of it underflows to 0


class MultiHeadAttention(Module):
    """``softmax(Q K^T / sqrt(d_k) + mask) V`` over ``n_heads`` heads, then an output projection.

    Used both as self-attention (``query_in is kv_in``) and as cross-attention
    (queries from the decoder, keys/values from the encoder memory).

    With ``n_kv_heads < n_heads`` it is grouped-query attention: only
    ``n_kv_heads`` key/value heads are projected, and each serves
    ``n_rep = n_heads // n_kv_heads`` query heads (``n_kv_heads = 1`` is
    multi-query attention). See notes/gqa.md.

    If ``rope`` is given, queries and keys are rotated by position after the
    head split; pass it for self-attention only.

    Shapes: query_in (batch, T_q, d_model), kv_in (batch, T_k, d_model),
    mask broadcastable to (batch, n_heads, T_q, T_k) -> (batch, T_q, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rope: RoPE | None = None,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per key/value head
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, n_kv_heads * self.d_k)
        self.w_v = Linear(d_model, n_kv_heads * self.d_k)
        self.w_o = Linear(d_model, d_model)
        self.attn_dropout = Dropout(dropout)
        self.register_modules(self.w_q, self.w_k, self.w_v, self.w_o, self.attn_dropout)
        self.rope = rope
        # attention probabilities of the last forward, (batch, n_heads, T_q, T_k), for plotting
        self.attn_weights: Array | None = None
        self._cache = None

    def _split_heads(self, x: Array, n_heads: int) -> Array:
        """(batch, T, n_heads * d_k) -> (batch, n_heads, T, d_k)."""
        b, t, _ = x.shape
        return x.reshape(b, t, n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Array) -> Array:
        """(batch, n_heads, T, d_k) -> (batch, T, n_heads * d_k)."""
        b, h, t, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * d_k)

    def _repeat_kv(self, x: Array) -> Array:
        """(batch, n_kv_heads, T, d_k) -> (batch, n_heads, T, d_k); query head j uses kv head j // n_rep."""
        return x if self.n_rep == 1 else xp.repeat(x, self.n_rep, axis=1)

    def _sum_kv_grads(self, dx: Array) -> Array:
        """Backward of ``_repeat_kv``: sum the gradients of each kv head's copies."""
        if self.n_rep == 1:
            return dx
        b, _, t, d_k = dx.shape
        return dx.reshape(b, self.n_kv_heads, self.n_rep, t, d_k).sum(axis=2)

    def forward(
        self, query_in: Array, kv_in: Array, mask: Array | None = None
    ) -> Array:
        q = self._split_heads(self.w_q.forward(query_in), self.n_heads)
        k = self._split_heads(self.w_k.forward(kv_in), self.n_kv_heads)
        v = self._split_heads(self.w_v.forward(kv_in), self.n_kv_heads)

        if self.rope is not None:
            q = self.rope.forward(q)
            k = self.rope.forward(k)

        k, v = self._repeat_kv(k), self._repeat_kv(v)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = xp.where(mask, scores, NEG_INF)

        attn = softmax(scores, axis=-1)
        attn_dropped = self.attn_dropout.forward(attn)
        context = attn_dropped @ v

        out = self.w_o.forward(self._merge_heads(context))
        self.attn_weights = attn
        # q and k are cached after RoPE: the score gradient is taken w.r.t. the rotated vectors
        self._cache = (q, k, v, attn, attn_dropped, mask)

        return out

    def backward(self, d_out: Array) -> tuple[Array, Array]:
        """Returns (d_query_in, d_kv_in).

        For self-attention the caller adds the two, since both inputs are the
        same tensor; for cross-attention d_kv_in is the gradient w.r.t. the
        encoder memory.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        q, k, v, attn, attn_dropped, mask = self._cache

        d_context = self._split_heads(self.w_o.backward(d_out), self.n_heads)
        d_attn_dropped = d_context @ v.transpose(0, 1, 3, 2)
        d_v = attn_dropped.transpose(0, 1, 3, 2) @ d_context

        d_attn = self.attn_dropout.backward(d_attn_dropped)
        d_scores = softmax_backward(attn, d_attn, axis=-1)

        if mask is not None:
            d_scores = xp.where(mask, d_scores, 0.0)

        d_scores = d_scores * self.scale

        d_q = d_scores @ k
        d_k = d_scores.transpose(0, 1, 3, 2) @ q
        d_k, d_v = self._sum_kv_grads(d_k), self._sum_kv_grads(d_v)

        if self.rope is not None:
            d_q = self.rope.backward(d_q)
            d_k = self.rope.backward(d_k)

        d_query_in = self.w_q.backward(self._merge_heads(d_q))
        d_kv_from_k = self.w_k.backward(self._merge_heads(d_k))
        d_kv_from_v = self.w_v.backward(self._merge_heads(d_v))

        return d_query_in, d_kv_from_k + d_kv_from_v
