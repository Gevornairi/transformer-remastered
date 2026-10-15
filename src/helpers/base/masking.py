"""Boolean attention masks: ``True`` means the query may attend to that key.

Masks are shaped to broadcast against attention scores of shape
``(batch, heads, query_len, key_len)``.
"""

from .backend import Array, xp


def padding_mask(ids: Array, pad_id: int) -> Array:
    """Hide PAD keys. ``ids`` (batch, T) -> mask (batch, 1, 1, T)."""
    return (ids != pad_id)[:, None, None, :]


def causal_mask(size: int) -> Array:
    """Hide future keys: query i sees keys 0..i. Returns (1, 1, size, size)."""
    return (xp.tril(xp.ones((size, size))) > 0)[None, None, :, :]


def decoder_mask(tgt_ids: Array, pad_id: int) -> Array:
    """Decoder self-attention mask: padding and causal. ``tgt_ids`` (batch, T) -> (batch, 1, T, T)."""
    return padding_mask(tgt_ids, pad_id) & causal_mask(tgt_ids.shape[1])
