"""Translate sentences with a trained checkpoint.

  python -m src.translate "The weather is nice today."   # one or more sentences as arguments
  python -m src.translate                                # interactive: type a sentence, get a translation
  python -m src.translate < input.txt > output.txt       # one sentence per line, translated in batches

The model's architecture is read from the arguments stored in the
checkpoint, so it always matches the run that produced it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .transformer import Transformer
from .utils.batching import SpecialIds
from .utils.checkpoint import load_checkpoint
from .utils.evaluation import translate_ids


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


def translate_texts(
    model: Transformer,
    tokenizer: Tokenizer,
    sentences: list[str],
    special_ids: SpecialIds,
    beam: int,
    alpha: float,
) -> list[str]:
    """Translate raw sentences; blank input lines give blank output lines."""
    ids = [
        # same truncation as training: room for EOS within the RoPE table
        tokenizer.encode(s, add_special_tokens=False).ids[: model.max_len - 1]
        for s in sentences
    ]
    out = translate_ids(model, ids, special_ids, beam=beam, alpha=alpha)
    return [tokenizer.decode(t) if s.strip() else "" for s, t in zip(sentences, out)]


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
    special_ids: SpecialIds = (meta["pad_id"], meta["bos_id"], meta["eos_id"])
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    model = load_model(Path(args.ckpt), meta["vocab_size"])

    def run(sentences: list[str]) -> None:
        for line in translate_texts(
            model, tokenizer, sentences, special_ids, args.beam, args.alpha
        ):
            print(line)

    if args.sentences:
        run(args.sentences)
    elif sys.stdin.isatty():
        print(f"{meta['src_lang']} -> {meta['tgt_lang']}, Ctrl-D to quit")
        while True:
            try:
                sentence = input(f"{meta['src_lang']}> ")
            except EOFError:
                print()
                break
            if sentence.strip():
                run([sentence])
    else:
        run([line.rstrip("\n") for line in sys.stdin])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
