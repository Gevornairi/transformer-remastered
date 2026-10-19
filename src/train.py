"""Train the from-scratch Transformer on the WMT14 subset built by prepare_wmt14.

  python -m src.train               # Transformer-base, paper hyper-parameters
  python -m src.train --resume      # continue from checkpoints/.../last.npz
  python -m src.train --eval-only   # BLEU on newstest2014 with the best checkpoint

The GPU does the math (MLX); a background thread builds and pads the next
batches on the CPU at the same time.

Every --eval-every updates: validation loss and greedy BLEU on the first
--bleu-sents sentences of newstest2013, a sample translation, and checkpoints
(last.npz always, best.npz when BLEU improves).
"""

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer

from .helpers.base.backend import asarray, materialize, seed
from .helpers.loss.label_smoothing_loss import LabelSmoothingLoss
from .helpers.optimisers.adamw import AdamW
from .helpers.optimisers.cosine_schedule import CosineSchedule
from .helpers.optimisers.noam_schedule import NoamSchedule
from .transformer import Transformer
from .utils.batching import SpecialIds
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.evaluation import bleu, translate_split, validation_loss
from .utils.parallel_split import ParallelSplit
from .utils.prefetcher import Prefetcher


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
            sample = translate_split(model, tokenizer, valid, [0], special_ids)[0]
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
