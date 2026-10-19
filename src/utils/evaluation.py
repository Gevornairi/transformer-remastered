"""Translation, validation loss and BLEU for a trained or training model."""

from collections.abc import Sequence

import numpy as np
import sacrebleu
from tokenizers import Tokenizer

from ..helpers.base.backend import asarray, to_numpy
from ..helpers.loss.label_smoothing_loss import LabelSmoothingLoss
from ..transformer import Transformer
from .batching import SpecialIds, collate, strip_special, token_batches
from .parallel_split import ParallelSplit


def translate_ids(
    model: Transformer,
    src_ids: Sequence[Sequence[int]],
    special_ids: SpecialIds,
    batch_size: int = 64,
    extra_len: int = 50,
    beam: int = 1,
    alpha: float = 0.6,
) -> list[list[int]]:
    """Translate source token ids (without BOS/EOS) to target token ids, in input order.

    Sources are sorted by length and decoded batch_size at a time, so each
    batch has little padding. Greedy if beam <= 1, else beam search. Each
    output may be up to ``extra_len`` tokens longer than its batch's longest
    source; outputs have BOS/EOS/PAD removed.
    """
    pad, bos, eos = special_ids
    order = np.argsort([len(s) for s in src_ids], kind="stable")
    outputs: list[list[int]] = [[] for _ in src_ids]
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        # same layout as collate's src: ids + EOS, right-padded with PAD
        width = max(len(src_ids[i]) for i in chunk) + 1
        src = np.full((len(chunk), width), pad, dtype=np.int32)
        for row, i in enumerate(chunk):
            n = len(src_ids[i])
            src[row, :n] = src_ids[i]
            src[row, n] = eos
        max_len = width + extra_len
        if beam <= 1:
            ys = model.greedy_decode(
                asarray(src), bos, eos, max_len=max_len, pad_id=pad
            )
        else:
            ys = model.beam_decode(
                asarray(src),
                bos,
                eos,
                beam=beam,
                alpha=alpha,
                max_len=max_len,
                pad_id=pad,
            )
        for row, i in zip(to_numpy(ys), chunk):
            outputs[i] = strip_special(row, pad, bos, eos)
    return outputs


def translate_split(
    model: Transformer,
    tokenizer: Tokenizer,
    split: ParallelSplit,
    idx,
    special_ids: SpecialIds,
    beam: int = 1,
    alpha: float = 0.6,
) -> list[str]:
    """Translate the source side of pairs ``idx`` of split; returns detokenized strings in idx order."""
    ids = translate_ids(
        model, [split.src_ids(i) for i in idx], special_ids, beam=beam, alpha=alpha
    )
    return [tokenizer.decode(t) for t in ids]


def bleu(
    model: Transformer,
    tokenizer: Tokenizer,
    split: ParallelSplit,
    refs: list[str],
    special_ids: SpecialIds,
    n: int | None,
    beam: int = 1,
    alpha: float = 0.6,
) -> float:
    """Corpus BLEU (sacrebleu) on the first n sentences of split, or all if n is None."""
    idx = np.arange(len(split) if n is None else min(n, len(split)))
    hyps = translate_split(
        model, tokenizer, split, idx, special_ids, beam=beam, alpha=alpha
    )
    return sacrebleu.corpus_bleu(hyps, [[refs[i] for i in idx]]).score


def validation_loss(
    model: Transformer,
    loss_fn: LabelSmoothingLoss,
    split: ParallelSplit,
    special_ids: SpecialIds,
    batch_tokens: int,
) -> float:
    """Label-smoothed loss per target token over the whole split, in eval mode."""
    was_training = model.training
    model.eval()
    total, n_tokens = 0.0, 0
    for idx in token_batches(split, batch_tokens, np.random.default_rng(0)):
        src, tgt_in, tgt_out = collate(split, idx, *special_ids)
        n = int((tgt_out != special_ids[0]).sum())
        loss, _ = loss_fn(
            model.forward(asarray(src), asarray(tgt_in)), asarray(tgt_out)
        )
        total += loss.item() * n
        n_tokens += n
    model.train(was_training)
    return total / n_tokens
