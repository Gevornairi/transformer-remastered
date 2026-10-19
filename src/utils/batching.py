"""Turning a ParallelSplit into padded training batches, and back from decoded ids."""

import numpy as np

from .parallel_split import ParallelSplit

SpecialIds = tuple[int, int, int]  # (pad, bos, eos)


def collate(split: ParallelSplit, idx, pad: int, bos: int, eos: int):
    """Build padded int32 arrays for pairs ``idx``.

    src = x + EOS;  tgt_in = BOS + y;  tgt_out = y + EOS; right-padded with PAD.
    Returns (src, tgt_in, tgt_out), shapes (n, S) and (n, T) twice.
    """
    n = len(idx)
    s_max = int(split.src_len[idx].max()) + 1
    t_max = int(split.tgt_len[idx].max()) + 1
    src = np.full((n, s_max), pad, dtype=np.int32)
    tgt_in = np.full((n, t_max), pad, dtype=np.int32)
    tgt_out = np.full((n, t_max), pad, dtype=np.int32)
    for row, i in enumerate(idx):
        s, t = split.src_ids(i), split.tgt_ids(i)
        src[row, : len(s)] = s
        src[row, len(s)] = eos
        tgt_in[row, 0] = bos
        tgt_in[row, 1 : len(t) + 1] = t
        tgt_out[row, : len(t)] = t
        tgt_out[row, len(t)] = eos
    return src, tgt_in, tgt_out


def token_batches(
    split: ParallelSplit, batch_tokens: int, rng: np.random.Generator, pool: int = 100
) -> list[list[int]]:
    """One epoch of length-bucketed batches whose padded size stays under batch_tokens.

    Shuffle, sort within pools of ~pool batches so each batch has similar
    lengths (little padding), then shuffle the batch order.
    """
    order = rng.permutation(len(split))
    longest = np.maximum(split.src_len, split.tgt_len) + 1
    approx_per_batch = max(1, batch_tokens // int(longest.mean()))
    batches = []
    for start in range(0, len(order), pool * approx_per_batch):
        chunk = order[start : start + pool * approx_per_batch]
        chunk = chunk[np.argsort(longest[chunk], kind="stable")]
        cur, cur_max = [], 0
        for i in chunk:
            new_max = max(cur_max, longest[i])
            if cur and new_max * (len(cur) + 1) > batch_tokens:
                batches.append(cur)
                cur, new_max = [], longest[i]
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def strip_special(ids, pad: int, bos: int, eos: int) -> list[int]:
    """Drop BOS and cut at the first EOS or PAD, leaving only content tokens."""
    out = []
    for t in ids:
        t = int(t)
        if t == eos or t == pad:
            break
        if t != bos:
            out.append(t)
    return out
