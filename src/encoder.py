"""Transformer encoder: embedding, a stack of encoder layers, final RMSNorm."""

from .helpers.base.backend import Array
from .helpers.base.module import Module
from .helpers.layers.dropout import Dropout
from .helpers.layers.embedding import Embedding
from .helpers.layers.encoder_layer import EncoderLayer
from .helpers.layers.rms_norm import RMSNorm
from .helpers.layers.rope import RoPE


class Encoder(Module):
    """Maps source token ids to the memory the decoder attends to.

    ``src_ids`` (batch, T_src) -> memory (batch, T_src, d_model). Position
    comes only from RoPE inside self-attention; nothing is added to the
    embeddings. The final norm is needed with pre-norm blocks, whose residual
    stream is never normalised otherwise.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.embed_dropout = Dropout(dropout)
        self.rope = RoPE(d_model // n_heads, max_len)  # shared by every layer
        self.layers = [
            EncoderLayer(d_model, n_heads, d_ff, dropout, self.rope, n_kv_heads)
            for _ in range(n_layers)
        ]
        self.final_norm = RMSNorm(d_model)
        self.register_modules(
            self.embed, self.embed_dropout, *self.layers, self.final_norm
        )

    def forward(self, src_ids: Array, src_mask: Array | None = None) -> Array:
        """``src_mask`` (batch, 1, 1, T_src) hides PAD tokens; see ``masking.padding_mask``."""
        self._src_ids = src_ids
        x = self.embed_dropout.forward(self.embed.forward(src_ids))
        for layer in self.layers:
            x = layer.forward(x, src_mask)

        x = self.final_norm.forward(x)
        return x

    def backward(self, d_memory: Array) -> None:
        """Backpropagate the memory gradient (summed over decoder layers) into every parameter."""
        d = self.final_norm.backward(d_memory)
        for layer in reversed(self.layers):
            d = layer.backward(d)

        self.embed.backward(self.embed_dropout.backward(d), self._src_ids)
