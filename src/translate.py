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

from tokenizers import Tokenizer

from .utils.batching import SpecialIds
from .utils.checkpoint import load_model
from .utils.evaluation import translate_texts


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
