"""Translate sentences with a trained checkpoint.

  python -m src.translate "The weather is nice today."   # one or more sentences as arguments
  python -m src.translate                                # interactive: type a sentence, get a translation
  python -m src.translate < input.txt                    # one sentence per line from stdin

The model's architecture is read from the arguments stored in the
checkpoint, so it always matches the run that produced it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .helpers.base.backend import asarray, to_numpy
from .train_wmt import load_checkpoint, strip_special
from .transformer import Transformer


def load_model(ckpt: Path, vocab_size: int) -> Transformer:
    """Build the Transformer described by the checkpoint's training args and load its weights."""
    with np.load(ckpt) as z:
        train_args = json.loads(z["args"].item())
    model = Transformer(
        vocab_size,
        vocab_size,
        d_model=train_args["d_model"],
        n_layers=train_args["layers"],
        n_heads=train_args["heads"],
        n_kv_heads=train_args["kv_heads"],
        d_ff=train_args["d_ff"],
        dropout=0.0,
        max_len=train_args["max_len"],
        share_embeddings=True,
    )
    load_checkpoint(ckpt, model)
    model.eval()
    return model


def translate(
    model: Transformer,
    tokenizer: Tokenizer,
    sentence: str,
    special_ids: tuple[int, int, int],
    beam: int,
    alpha: float,
) -> str:
    """Translate one sentence (greedy if beam == 1, else beam search)."""
    pad, bos, eos = special_ids
    ids = tokenizer.encode(sentence, add_special_tokens=False).ids
    ids = ids[: model.max_len - 1] + [eos]  # same truncation as training
    src = asarray([ids])
    max_len = len(ids) + 50
    if beam <= 1:
        out = model.greedy_decode(src, bos, eos, max_len=max_len, pad_id=pad)
    else:
        out = model.beam_decode(
            src, bos, eos, beam=beam, alpha=alpha, max_len=max_len, pad_id=pad
        )
    return tokenizer.decode(strip_special(to_numpy(out)[0], pad, bos, eos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences", nargs="*", help="omit to read from stdin")
    parser.add_argument("--data", default="data/wmt14_en_de")
    parser.add_argument("--ckpt", default="checkpoints/wmt14_en_de/best.npz")
    parser.add_argument("--beam", type=int, default=4, help="1 = greedy")
    parser.add_argument("--alpha", type=float, default=0.6, help="beam length penalty")
    args = parser.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text())
    special_ids = (meta["pad_id"], meta["bos_id"], meta["eos_id"])
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    model = load_model(Path(args.ckpt), meta["vocab_size"])

    def run(sentence: str) -> None:
        if sentence.strip():
            print(
                translate(
                    model, tokenizer, sentence, special_ids, args.beam, args.alpha
                )
            )

    if args.sentences:
        for sentence in args.sentences:
            run(sentence)
    elif sys.stdin.isatty():
        print(f"{meta['src_lang']} -> {meta['tgt_lang']}, Ctrl-D to quit")
        while True:
            try:
                sentence = input(f"{meta['src_lang']}> ")
            except EOFError:
                print()
                break
            run(sentence)
    else:
        for line in sys.stdin:
            run(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
