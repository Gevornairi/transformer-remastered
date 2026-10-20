"""Encoder-decoder Transformer for translation, with greedy and beam-search decoding."""

from nnkit.activations import log_softmax, softcap, softcap_backward
from nnkit.base import INT, Array, Module, decoder_mask, materialize, padding_mask, xp
from nnkit.layers import Embedding, Linear

from .decoder import Decoder
from .encoder import Encoder

PAD_ID = 0


class Transformer(Module):
    """The full model: encoder, decoder, and the projection to vocabulary logits.

    Compared with the 2017 paper: pre-norm blocks with RMSNorm, no biases,
    SwiGLU feed-forward, RoPE instead of sinusoidal position vectors, and
    optional grouped-query attention (``n_kv_heads``).

    With ``share_embeddings`` (requires a joint vocabulary) one table serves
    as source embedding, target embedding and output projection. Otherwise
    the target gets its own embedding and the output its own ``Linear``.

    With ``logit_cap > 0`` the logits are soft-capped, ``cap * tanh(logits / cap)``
    (Gemma 2), in ``project``, so training and both decoders see the same logits.

    Training: ``logits = forward(src, tgt_in)``, get ``d_logits`` from the
    loss, then ``backward(d_logits)`` fills every ``Param.grad``.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1344,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = True,
        n_kv_heads: int | None = None,
        logit_cap: float = 0.0,
    ) -> None:
        """
        Args:
            d_ff: SwiGLU hidden size; 1344 ~ 8/3 * 512 matches the paper's FFN parameter count.
            max_len: longest sequence the RoPE tables cover.
            n_kv_heads: key/value heads for grouped-query attention; None means n_heads.
            logit_cap: soft-cap on the output logits; 0 disables it.
        """
        super().__init__()
        if logit_cap < 0:
            raise ValueError(f"logit_cap={logit_cap} must be >= 0 (0 disables it)")
        self.d_model = d_model
        self.max_len = max_len
        self.share_embeddings = share_embeddings
        self.logit_cap = logit_cap
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            n_layers,
            n_heads,
            d_ff,
            dropout,
            max_len,
            n_kv_heads,
        )
        if share_embeddings:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("share_embeddings needs a joint vocabulary")
            tgt_embed = self.encoder.embed
        else:
            tgt_embed = Embedding(tgt_vocab_size, d_model)

        self.decoder = Decoder(
            tgt_embed, d_model, n_layers, n_heads, d_ff, dropout, max_len, n_kv_heads
        )
        self.register_modules(self.encoder, self.decoder)

        if share_embeddings:
            self.out_proj = None
            self._tied_weight = tgt_embed.weight
        else:
            self.out_proj = Linear(d_model, tgt_vocab_size)
            self.register_modules(self.out_proj)
        self._out_proj_input: Array | None = None
        self._capped_logits: Array | None = None

    def project(self, x: Array) -> Array:
        """Decoder states (..., d_model) -> logits (..., tgt_vocab).

        With shared embeddings this is ``x @ E^T``, where E is the embedding
        table (without the sqrt(d_model) scale the embedding lookup applies).
        With ``logit_cap`` the result is then soft-capped.
        """
        if self.out_proj is not None:
            logits = self.out_proj.forward(x)
        else:
            self._out_proj_input = x
            logits = x @ self._tied_weight.value.T

        if self.logit_cap:
            logits = softcap(logits, self.logit_cap)
            self._capped_logits = logits

        return logits

    def project_backward(self, d_logits: Array) -> Array:
        """Backward of ``project``: adds into the (possibly tied) weight's grad, returns dx."""
        if self.logit_cap:
            if self._capped_logits is None:
                raise RuntimeError("project_backward() called before project()")
            d_logits = softcap_backward(self._capped_logits, d_logits, self.logit_cap)

        if self.out_proj is not None:
            return self.out_proj.backward(d_logits)

        x = self._out_proj_input
        if x is None:
            raise RuntimeError("project_backward() called before project()")
        d_model = x.shape[-1]
        vocab = d_logits.shape[-1]
        self._tied_weight.grad += d_logits.reshape(-1, vocab).T @ x.reshape(-1, d_model)

        return d_logits @ self._tied_weight.value

    def forward(self, src_ids: Array, tgt_in_ids: Array, pad_id: int = PAD_ID) -> Array:
        """Teacher-forced logits.

        Args:
            src_ids: (batch, T_src) source ids, PAD-padded on the right.
            tgt_in_ids: (batch, T_tgt) target ids shifted right (BOS first).

        Returns:
            (batch, T_tgt, tgt_vocab) logits; position t predicts target token t.
        """
        src_mask = padding_mask(src_ids, pad_id)
        tgt_mask = decoder_mask(tgt_in_ids, pad_id)
        memory = self.encoder.forward(src_ids, src_mask)
        dec_out = self.decoder.forward(tgt_in_ids, memory, src_mask, tgt_mask)

        return self.project(dec_out)

    def backward(self, d_logits: Array) -> None:
        """Backpropagate ``d_logits`` (batch, T_tgt, vocab) from the last ``forward`` into every ``Param.grad``."""
        d_dec_out = self.project_backward(d_logits)
        d_memory = self.decoder.backward(d_dec_out)
        self.encoder.backward(d_memory)

    def greedy_decode(
        self,
        src_ids: Array,
        bos_id: int,
        eos_id: int,
        max_len: int = 64,
        pad_id: int = PAD_ID,
    ) -> Array:
        """Translate by taking the most likely token at every step.

        Returns (batch, <=max_len) ids starting with BOS; positions after EOS
        are PAD. Runs in eval mode and restores the previous mode afterwards.
        The whole prefix is re-decoded each step (no KV cache).
        """
        max_len = min(max_len, self.max_len)
        was_training = self.training
        self.eval()
        try:
            src_mask = padding_mask(src_ids, pad_id)
            memory = self.encoder.forward(src_ids, src_mask)
            batch = src_ids.shape[0]
            ys = xp.full((batch, 1), bos_id, dtype=INT)
            done = xp.zeros((batch,), dtype=xp.bool_)

            for _ in range(max_len - 1):
                tgt_mask = decoder_mask(ys, pad_id)
                dec_out = self.decoder.forward(ys, memory, src_mask, tgt_mask)
                logits = self.project(dec_out[:, -1, :])
                next_ids = logits.argmax(axis=-1).astype(INT)
                next_ids = xp.where(done, pad_id, next_ids)
                ys = xp.concatenate([ys, next_ids[:, None]], axis=1)
                done = done | (next_ids == eos_id)
                materialize(ys, done)
                if done.all().item():
                    break
            return ys
        finally:
            self.train(was_training)

    def beam_decode(
        self,
        src_ids: Array,
        bos_id: int,
        eos_id: int,
        beam: int = 4,
        alpha: float = 0.6,
        max_len: int = 64,
        pad_id: int = PAD_ID,
    ) -> Array:
        """Translate with beam search, ranking hypotheses by log P / ((5 + len) / 6)^alpha.

        The length penalty (Wu et al., 2016) stops the search from favouring
        short outputs; ``beam=4, alpha=0.6`` are the paper's settings.

        Returns (batch, <=max_len) ids starting with BOS; positions after EOS
        are PAD. Runs in eval mode and restores the previous mode afterwards.
        """
        max_len = min(max_len, self.max_len)
        was_training = self.training
        self.eval()
        try:
            b, k = src_ids.shape[0], beam
            src_mask = padding_mask(src_ids, pad_id)
            memory = self.encoder.forward(src_ids, src_mask)
            # row i*k + j is beam j of sentence i; encode once, then copy
            memory = xp.repeat(memory, k, axis=0)
            src_mask = xp.repeat(src_mask, k, axis=0)

            ys = xp.full((b * k, 1), bos_id, dtype=INT)
            # only beam 0 is live at the start, so step 1 doesn't pick k copies of one token
            score = xp.zeros((b, k)) + xp.where(
                xp.arange(k) == 0, 0.0, -1e9
            )  # sum log P
            length = xp.zeros((b, k))  # tokens generated, EOS included
            done = xp.zeros((b, k), dtype=xp.bool_)

            def length_penalty(n):
                return ((5.0 + n) / 6.0) ** alpha

            for _ in range(max_len - 1):
                dec_out = self.decoder.forward(
                    ys, memory, src_mask, decoder_mask(ys, pad_id)
                )
                logp = log_softmax(self.project(dec_out[:, -1, :]), axis=-1)
                vocab = logp.shape[-1]
                logp = logp.reshape(b, k, vocab)
                # a finished beam has one continuation: PAD, at no cost
                pad_only = xp.where(xp.arange(vocab) == pad_id, 0.0, -1e9)
                logp = xp.where(done[:, :, None], pad_only, logp)

                cand_score = score[:, :, None] + logp  # (b, k, vocab)
                cand_len = length + xp.where(done, 0.0, 1.0)  # (b, k)
                rank = (cand_score / length_penalty(cand_len)[:, :, None]).reshape(
                    b, k * vocab
                )
                top = xp.argpartition(-rank, k - 1, axis=-1)[:, :k]  # (b, k), unordered
                from_beam, tok = top // vocab, top % vocab

                score = xp.take_along_axis(
                    cand_score.reshape(b, k * vocab), top, axis=-1
                )
                length = xp.take_along_axis(cand_len, from_beam, axis=-1)
                done = xp.take_along_axis(done, from_beam, axis=-1) | (tok == eos_id)
                rows = (xp.arange(b)[:, None] * k + from_beam).reshape(-1)
                ys = xp.concatenate([ys[rows], tok.reshape(-1, 1).astype(INT)], axis=1)
                materialize(ys, score, length, done)
                if done.all().item():
                    break

            best = (score / length_penalty(length)).argmax(axis=-1)
            return ys[xp.arange(b) * k + best]
        finally:
            self.train(was_training)

    def attention_maps(self) -> dict[str, list[Array]]:
        """Attention weights from the most recent forward pass, (batch, heads, q, k) per layer."""
        maps = {
            "encoder_self": [l.self_attn.attn_weights for l in self.encoder.layers],
            "decoder_self": [l.self_attn.attn_weights for l in self.decoder.layers],
            "decoder_cross": [l.cross_attn.attn_weights for l in self.decoder.layers],
        }
        if any(w is None for ws in maps.values() for w in ws):
            raise RuntimeError("attention_maps() called before a forward pass")

        return maps  # type: ignore[return-value]

    def n_params(self) -> int:
        """Number of trainable scalars; a shared table is counted once."""
        return int(sum(p.value.size for p in self.parameters()))
