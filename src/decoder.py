"""Transformer decoder: embedding, a stack of decoder layers, final RMSNorm."""

from .helpers.base.backend import Array
from .helpers.base.module import Module
from .helpers.layers.decoder_layer import DecoderLayer
from .helpers.layers.dropout import Dropout
from .helpers.layers.embedding import Embedding
from .helpers.layers.rms_norm import RMSNorm
from .helpers.layers.rope import RoPE


class Decoder(Module):
    """Maps target-prefix ids plus the encoder memory to hidden states.

    ``tgt_ids`` (batch, T_tgt), memory (batch, T_src, d_model)
    -> (batch, T_tgt, d_model). The projection to vocabulary logits lives in
    ``Transformer.project`` because it may share the embedding table.

    ``embed`` is passed in rather than built here so the encoder's table can
    be reused when source and target share a vocabulary.
    """

    def __init__(
        self,
        embed: Embedding,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.embed_dropout = Dropout(dropout)
        self.rope = RoPE(d_model // n_heads, max_len)  # shared by every layer
        self.layers = [
            DecoderLayer(d_model, n_heads, d_ff, dropout, self.rope, n_kv_heads)
            for _ in range(n_layers)
        ]
        self.final_norm = RMSNorm(d_model)
        self.register_modules(
            self.embed, self.embed_dropout, *self.layers, self.final_norm
        )

    def forward(
        self,
        tgt_ids: Array,
        memory: Array,
        src_mask: Array | None = None,
        tgt_mask: Array | None = None,
    ) -> Array:
        """``src_mask`` (batch, 1, 1, T_src); ``tgt_mask`` (batch, 1, T_tgt, T_tgt), see ``masking``."""
        self._tgt_ids = tgt_ids
        x = self.embed_dropout.forward(self.embed.forward(tgt_ids))
        for layer in self.layers:
            x = layer.forward(x, memory, src_mask, tgt_mask)

        x = self.final_norm.forward(x)
        return x

    def backward(self, dy: Array) -> Array:
        """Backpropagate into every decoder parameter; returns the gradient w.r.t. memory.

        Every layer cross-attends to the same memory, so their memory
        gradients are summed.
        """
        d = self.final_norm.backward(dy)
        d_memory_total = None

        for layer in reversed(self.layers):
            d, d_memory = layer.backward(d)
            d_memory_total = (
                d_memory if d_memory_total is None else d_memory_total + d_memory
            )

        self.embed.backward(self.embed_dropout.backward(d), self._tgt_ids)

        return d_memory_total
