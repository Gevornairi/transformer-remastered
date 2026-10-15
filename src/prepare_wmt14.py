"""Download WMT14, sample a training subset, train a shared BPE and pre-tokenize.

  python -m src.prepare_wmt14 --n-train 1000000 --vocab-size 32000

Writes to --out:
  tokenizer.json          shared source/target BPE (PAD=0, BOS=1, EOS=2, UNK=3)
  {train,valid,test}.npz  flat int32 token ids + int64 offsets per side (no BOS/EOS)
  {valid,test}.{src,tgt}  raw text, one sentence per line, for BLEU
  meta.json               vocab size, special ids, counts, settings

valid is newstest2013 and test is newstest2014, kept whole so BLEU is on the
official sets; only train pairs are length-filtered.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

SPECIALS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
]  # ids 0..3, matching PAD/BOS/EOS in training


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    """Train a byte-pair-encoding tokenizer (NFKC, Metaspace) on texts from both languages."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    for i, s in enumerate(SPECIALS):
        assert tokenizer.token_to_id(s) == i, f"{s} should have id {i}"
    return tokenizer


def encode(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Token ids per text, without special tokens."""
    # encode_batch fans out over all CPU cores in Rust
    return [e.ids for e in tokenizer.encode_batch(texts, add_special_tokens=False)]


def pack(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ragged sequences: returns (flat int32 ids, int64 offsets of length n + 1)."""
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    offsets = np.concatenate([[0], np.cumsum(lens)])
    flat = np.fromiter(
        (t for s in seqs for t in s), dtype=np.int32, count=int(offsets[-1])
    )
    return flat, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/wmt14_en_de")
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="de")
    parser.add_argument(
        "--n-train", type=int, default=1_000_000, help="pairs sampled from 4.5M"
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="drop longer train pairs"
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=3.0,
        help="drop train pairs with a worse length ratio",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    non_english = args.tgt if args.src == "en" else args.src
    # hub configs are named "<xx>-en"
    dataset = load_dataset("wmt/wmt14", f"{non_english}-en")
    train = (
        dataset["train"]
        .shuffle(seed=args.seed)
        .select(range(min(args.n_train, len(dataset["train"]))))
    )

    def split_sides(split) -> tuple[list[str], list[str]]:
        """(source sentences, target sentences) of a hub split."""
        pairs = split["translation"]
        return [p[args.src].strip() for p in pairs], [
            p[args.tgt].strip() for p in pairs
        ]

    train_src, train_tgt = split_sides(train)
    print(f"loaded {len(train_src):,} train pairs in {time.time() - t0:.0f}s")

    tokenizer = build_tokenizer(train_src + train_tgt, args.vocab_size)
    tokenizer.save(str(out / "tokenizer.json"))
    print(
        f"trained BPE ({tokenizer.get_vocab_size():,} types) at {time.time() - t0:.0f}s"
    )

    counts = {}
    for name, split in (
        ("train", None),
        ("valid", dataset["validation"]),
        ("test", dataset["test"]),
    ):
        if split is None:
            src_text, tgt_text = train_src, train_tgt
        else:
            src_text, tgt_text = split_sides(split)
        src_ids, tgt_ids = encode(tokenizer, src_text), encode(tokenizer, tgt_text)

        if name == "train":
            keep = [
                i
                for i, (s, t) in enumerate(zip(src_ids, tgt_ids))
                if 0 < len(s) <= args.max_tokens
                and 0 < len(t) <= args.max_tokens
                and max(len(s), len(t)) <= args.max_ratio * min(len(s), len(t))
            ]
            src_ids = [src_ids[i] for i in keep]
            tgt_ids = [tgt_ids[i] for i in keep]
        else:
            # keep every eval sentence so BLEU is on the full official set
            (out / f"{name}.src").write_text("\n".join(src_text) + "\n")
            (out / f"{name}.tgt").write_text("\n".join(tgt_text) + "\n")

        src_flat, src_off = pack(src_ids)
        tgt_flat, tgt_off = pack(tgt_ids)
        np.savez(
            out / f"{name}.npz",
            src=src_flat,
            src_off=src_off,
            tgt=tgt_flat,
            tgt_off=tgt_off,
        )
        counts[name] = {
            "pairs": len(src_ids),
            "src_tokens": int(src_off[-1]),
            "tgt_tokens": int(tgt_off[-1]),
        }
        print(f"{name}: {counts[name]}")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "src_lang": args.src,
        "tgt_lang": args.tgt,
        "counts": counts,
        "args": vars(args),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
