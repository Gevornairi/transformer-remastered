"""Train the from-scratch Transformer on the WMT14 subset built by prepare_wmt14.

  python -m src.train_wmt                      # Transformer-base, paper hyper-parameters
  python -m src.train_wmt --resume             # continue from checkpoints/.../last.npz
  python -m src.train_wmt --eval-only --resume # BLEU on newstest2014 with the best checkpoint

The GPU does the math (MLX); a background thread builds and pads the next
batches on the CPU at the same time.

Every --eval-every updates: validation loss and greedy BLEU on the first
--bleu-sents sentences of newstest2013, a sample translation, and checkpoints
(last.npz always, best.npz when BLEU improves).
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sacrebleu
from tokenizers import Tokenizer

from .helpers.base.backend import asarray, materialize, seed, to_numpy
from .helpers.loss.label_smoothing_loss import LabelSmoothingLoss
from .helpers.optimisers.adamw import AdamW
from .helpers.optimisers.cosine_schedule import CosineSchedule
from .helpers.optimisers.noam_schedule import NoamSchedule
from .transformer import Transformer

SpecialIds = tuple[int, int, int]  # (pad, bos, eos)


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


class Prefetcher:
    """Builds padded numpy batches on a background thread, forever (epoch after epoch)."""

    def __init__(
        self,
        split: ParallelSplit,
        batch_tokens: int,
        special_ids: SpecialIds,
        seed_: int,
        depth: int = 16,
    ) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.epoch = 0
        args = (split, batch_tokens, special_ids, np.random.default_rng(seed_))
        threading.Thread(target=self._run, args=args, daemon=True).start()

    def _run(self, split, batch_tokens, special_ids, rng):
        while True:
            for idx in token_batches(split, batch_tokens, rng):
                self.queue.put((self.epoch, collate(split, idx, *special_ids)))
            self.epoch += 1

    def next(self):
        """Blocks until a batch is ready; returns (epoch, (src, tgt_in, tgt_out))."""
        return self.queue.get()


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


def translate(
    model: Transformer,
    tokenizer: Tokenizer,
    split: ParallelSplit,
    idx,
    special_ids: SpecialIds,
    batch_size: int = 64,
    extra_len: int = 50,
    beam: int = 1,
    alpha: float = 0.6,
) -> list[str]:
    """Decode sentences idx of split (greedy, or beam search if beam > 1).

    Each output may be up to ``extra_len`` tokens longer than its batch's
    longest source. Returns detokenized strings in idx order.
    """
    pad, bos, eos = special_ids
    idx = np.asarray(idx)
    order = idx[np.argsort(split.src_len[idx], kind="stable")]  # less padding
    hyps: dict[int, str] = {}
    for b in range(0, len(order), batch_size):
        chunk = order[b : b + batch_size]
        src, _, _ = collate(split, chunk, pad, bos, eos)
        out = to_numpy(
            model.greedy_decode(
                asarray(src), bos, eos, max_len=src.shape[1] + extra_len, pad_id=pad
            )
            if beam <= 1
            else model.beam_decode(
                asarray(src),
                bos,
                eos,
                beam=beam,
                alpha=alpha,
                max_len=src.shape[1] + extra_len,
                pad_id=pad,
            )
        )
        for row, i in zip(out, chunk):
            hyps[int(i)] = tokenizer.decode(strip_special(row, pad, bos, eos))
    return [hyps[int(i)] for i in idx]


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
    hyps = translate(model, tokenizer, split, idx, special_ids, beam=beam, alpha=alpha)
    return sacrebleu.corpus_bleu(hyps, [[refs[i] for i in idx]]).score


def save_checkpoint(
    path: Path, model: Transformer, opt: AdamW, step: int, best_bleu: float, args
) -> None:
    """Write model, optimizer state, progress and CLI args to path (atomically, via a temp file)."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        **model.state(),
        **{f"opt_{k}": v for k, v in opt.state().items()},
        step=step,
        best_bleu=best_bleu,
        args=json.dumps(vars(args)),
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path, model: Transformer, opt: AdamW | None = None
) -> tuple[int, float]:
    """Restore model (and optimizer, if given) from path; returns (step, best_bleu)."""
    z = np.load(path)
    model.load_state({k: z[k] for k in z.files if k.startswith("p")})
    if opt is not None:
        opt.load_state({k[4:]: z[k] for k in z.files if k.startswith("opt_")})
    return int(z["step"]), float(z["best_bleu"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wmt14_en_de")
    parser.add_argument("--ckpt-dir", default="checkpoints/wmt14_en_de")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="test-set BLEU with best.npz, then exit",
    )
    # model (defaults = Transformer-base from the paper)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--kv-heads",
        type=int,
        default=None,
        help="GQA key/value heads; default = --heads",
    )
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=256, help="RoPE table size")
    # optimisation
    parser.add_argument(
        "--steps",
        type=int,
        default=20_000,
        help="optimizer updates (~17 epochs of 1M pairs)",
    )
    parser.add_argument(
        "--batch-tokens", type=int, default=4096, help="padded tokens per micro-batch"
    )
    parser.add_argument(
        "--accum",
        type=int,
        default=6,
        help="micro-batches per update (~25k tokens like the paper)",
    )
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--schedule", choices=["cosine", "noam"], default="cosine")
    parser.add_argument(
        "--lr",
        type=float,
        default=7e-4,
        help="cosine peak lr (Noam's peak for d_model=512, warmup=4000)",
    )
    parser.add_argument("--min-lr", type=float, default=7e-5, help="cosine final lr")
    parser.add_argument("--lr-factor", type=float, default=1.0, help="noam only")
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="AdamW, weight matrices only"
    )
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0, help="0 disables")
    parser.add_argument("--seed", type=int, default=0)
    # monitoring
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument(
        "--bleu-sents",
        type=int,
        default=500,
        help="newstest2013 sentences for periodic BLEU",
    )
    parser.add_argument(
        "--beam", type=int, default=4, help="--eval-only beam size; 1 = greedy only"
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="beam length penalty")
    args = parser.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text())
    special_ids: SpecialIds = (meta["pad_id"], meta["bos_id"], meta["eos_id"])
    pad = special_ids[0]
    vocab_size = meta["vocab_size"]
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    # every split is truncated so BOS/EOS + tokens fit the RoPE table
    max_tokens = args.max_len - 1
    valid = ParallelSplit(data / "valid.npz", max_tokens)
    valid_refs = (data / "valid.tgt").read_text().splitlines()

    seed(args.seed)
    model = Transformer(
        vocab_size,
        vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        share_embeddings=True,
    )
    loss_fn = LabelSmoothingLoss(vocab_size, pad, args.smoothing)
    if args.schedule == "cosine":
        # depends on --steps: resume with the same value
        schedule = CosineSchedule(args.lr, args.warmup, args.steps, args.min_lr)
    else:
        schedule = NoamSchedule(args.d_model, warmup=args.warmup, factor=args.lr_factor)
    opt = AdamW(
        model.parameters(),
        schedule,
        clip_norm=args.clip_norm or None,
        weight_decay=args.weight_decay,
    )
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        load_checkpoint(ckpt_dir / "best.npz", model)
        test = ParallelSplit(data / "test.npz", max_tokens)
        refs = (data / "test.tgt").read_text().splitlines()
        t0 = time.time()
        greedy = bleu(model, tokenizer, test, refs, special_ids, None)
        msg = (
            f"newstest2014 {meta['src_lang']}->{meta['tgt_lang']} "
            f"greedy BLEU {greedy:.2f} ({time.time() - t0:.0f}s)"
        )
        if args.beam > 1:
            t0 = time.time()
            beam = bleu(
                model,
                tokenizer,
                test,
                refs,
                special_ids,
                None,
                beam=args.beam,
                alpha=args.alpha,
            )
            msg += f" | beam{args.beam} BLEU {beam:.2f} ({time.time() - t0:.0f}s)"
        print(msg)
        return 0

    step, best_bleu = 0, -1.0
    if args.resume and (ckpt_dir / "last.npz").exists():
        step, best_bleu = load_checkpoint(ckpt_dir / "last.npz", model, opt)
        print(f"resumed at step {step} (best BLEU {best_bleu:.2f})")

    train = ParallelSplit(data / "train.npz", max_tokens)
    print(
        f"{model.n_params() / 1e6:.1f}M params | {len(train):,} train pairs | "
        f"~{args.batch_tokens * args.accum:,} padded tokens/update"
    )
    batches = Prefetcher(train, args.batch_tokens, special_ids, args.seed + step)

    # running totals since the last log line
    log_t0, log_tokens, log_loss_sum = time.time(), 0, 0.0
    epoch = 0
    model.train()
    while step < args.steps:
        model.zero_grad()
        for _ in range(args.accum):
            epoch, (src, tgt_in, tgt_out) = batches.next()
            n_tgt = int((tgt_out != pad).sum())
            logits = model.forward(asarray(src), asarray(tgt_in))
            loss, d_logits = loss_fn(logits, asarray(tgt_out))
            model.backward(d_logits * (1.0 / args.accum))
            materialize(
                loss, [p.grad for p in model.parameters()]
            )  # bound the lazy graph
            log_loss_sum += loss.item() * n_tgt
            log_tokens += n_tgt
        grad_norm = opt.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - log_t0
            print(
                f"step {step:6d} | ep {epoch} | loss {log_loss_sum / log_tokens:.3f} | "
                f"lr {opt.last_lr:.2e} | gnorm {grad_norm.item():.2f} | "
                f"{log_tokens / dt:,.0f} tgt tok/s | {dt / args.log_every:.2f}s/step",
                flush=True,
            )
            log_t0, log_tokens, log_loss_sum = time.time(), 0, 0.0

        if step % args.eval_every == 0 or step == args.steps:
            t0 = time.time()
            v_loss = validation_loss(
                model, loss_fn, valid, special_ids, args.batch_tokens
            )
            score = bleu(
                model, tokenizer, valid, valid_refs, special_ids, args.bleu_sents
            )
            is_best = score > best_bleu
            best_bleu = max(best_bleu, score)
            save_checkpoint(ckpt_dir / "last.npz", model, opt, step, best_bleu, args)
            if is_best:
                save_checkpoint(
                    ckpt_dir / "best.npz", model, opt, step, best_bleu, args
                )
            sample = translate(model, tokenizer, valid, [0], special_ids)[0]
            print(
                f"== eval step {step}: valid loss {v_loss:.3f} | BLEU@{args.bleu_sents} {score:.2f}"
                f"{' (best)' if is_best else ''} | {time.time() - t0:.0f}s\n"
                f"   src: {(data / 'valid.src').read_text().splitlines()[0]}\n"
                f"   hyp: {sample}\n   ref: {valid_refs[0]}",
                flush=True,
            )
            log_t0 = time.time()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
