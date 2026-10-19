"""Pre-tokenized sentence pairs loaded from a ``{train,valid,test}.npz`` file."""

from pathlib import Path

import numpy as np


class ParallelSplit:
    """A pre-tokenized split of sentence pairs: ragged src/tgt id lists stored flat with offsets.

    Pair i's source ids are ``src[src_off[i] : src_off[i + 1]]`` (same for the
    target). With ``max_tokens``, each side is truncated to that many tokens
    so that, after adding BOS/EOS, it still fits the model's RoPE table.
    """

    def __init__(self, path: Path, max_tokens: int | None = None) -> None:
        z = np.load(path)
        self.src, self.src_off = z["src"], z["src_off"]
        self.tgt, self.tgt_off = z["tgt"], z["tgt_off"]
        self.src_len = np.diff(self.src_off)
        self.tgt_len = np.diff(self.tgt_off)
        if max_tokens is not None:
            self.src_len = np.minimum(self.src_len, max_tokens)
            self.tgt_len = np.minimum(self.tgt_len, max_tokens)

    def __len__(self) -> int:
        return len(self.src_len)

    def src_ids(self, i: int) -> np.ndarray:
        return self.src[self.src_off[i] : self.src_off[i] + self.src_len[i]]

    def tgt_ids(self, i: int) -> np.ndarray:
        return self.tgt[self.tgt_off[i] : self.tgt_off[i] + self.tgt_len[i]]
