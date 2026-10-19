# transformer-remastered

An encoder-decoder Transformer for English→German translation (WMT14), written from scratch.
There is no autograd: every layer has a hand-written `forward` and `backward`, and a gradient check verifies them.
The model starts from the 2017 architecture ("Attention Is All You Need") and replaces its parts with the ones modern models use.

| 2017 paper | This repo |
|---|---|
| Post-LN residual blocks | Pre-LN, plus a final norm after the encoder and decoder stacks |
| LayerNorm | RMSNorm |
| Biases in every linear layer | No biases |
| ReLU feed-forward, `d_ff = 4·d` | SwiGLU, `d_ff ≈ 8/3·d` (1344 for d = 512), same parameter count |
| Sinusoidal position vectors added to embeddings | RoPE in self-attention |
| Multi-head attention | Optional grouped-query / multi-query attention (`n_kv_heads`) |
| Adam + Noam schedule | AdamW (decay on weight matrices only) + warmup and cosine decay; Noam is still available |
| Greedy decoding | Beam search with length penalty (beam 4, α = 0.6) |

Kept from the paper: tied source/target/output embeddings with a joint BPE vocabulary, label smoothing 0.1, ~25k tokens per update, Transformer-base size (d = 512, 6 + 6 layers, 8 heads).

The layers, losses and optimisers come from [nnkit](https://github.com/Gevornairi/nnkit), a separate from-scratch toolkit; this repo holds the Transformer model, the WMT14 data pipeline, training and evaluation.
All math goes through one array module, so the same code runs on the Apple GPU (MLX) or the CPU (NumPy).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). MLX needs Apple silicon; elsewhere, use the NumPy backend.

```sh
uv sync
```

This also installs nnkit at the version pinned in `pyproject.toml` (a git tag).

To develop nnkit alongside this repo, clone it next to this one and temporarily point the dependency at your local copy:

```toml
# pyproject.toml, [tool.uv.sources]
nnkit = { path = "../nnkit", editable = true }
```

Before committing here, tag a new nnkit release and switch back to `{ git = "https://github.com/Gevornairi/nnkit", tag = "vX.Y.Z" }`.

## Usage

```sh
# 1. Data: download WMT14 en-de, sample 1M training pairs, train a 32k shared BPE, pre-tokenize
uv run python -m src.prepare_wmt14 --n-train 1000000 --vocab-size 32000

# 2. Train (Transformer-base defaults; checkpoints go to checkpoints/wmt14_en_de/)
uv run python -m src.train
uv run python -m src.train --resume            # continue from last.npz

# 3. Test-set BLEU on newstest2014 (greedy and beam) with best.npz
uv run python -m src.train --eval-only

# 4. Translate your own sentences with best.npz (beam 4; --beam 1 for greedy)
uv run python -m src.translate "The weather is nice today."
uv run python -m src.translate                     # interactive, Ctrl-D to quit
```

`translate.py` rebuilds the model from the settings stored in the checkpoint, so it always matches the training run.

`uv run python -m src.train --help` lists every option. Useful ones:
`--kv-heads` (GQA), `--schedule noam`, `--steps`, `--batch-tokens` / `--accum` (tokens per update), `--max-len` (longest sequence; longer sentences are truncated).
With the cosine schedule, pass the same `--steps` when resuming, since the schedule depends on it.

## Gradient check

```sh
# exact: central finite differences in float64 on the CPU
NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 uv run python -m tests.gradcheck

# MLX autodiff of the same forward pass, float32 on the GPU
uv run python -m tests.gradcheck
```

It checks a tiny model with shared and separate embeddings, and with multi-head, grouped-query and multi-query attention.
It exits non-zero if any gradient is off.

## Backend

Chosen with environment variables, read by nnkit when it is first imported:

| Variable | Values | Default |
|---|---|---|
| `NNKIT_BACKEND` | `mlx`, `numpy` | `mlx` |
| `NNKIT_DTYPE` | `float32`, `float64` (NumPy only) | `float32` |

## Layout

```
src/
  transformer.py          Transformer: encoder + decoder + output projection, greedy/beam decoding
  encoder.py, decoder.py  embedding → stack of layers → final RMSNorm
  train.py                training loop: arguments, model and optimizer setup, logging, periodic eval
  prepare_wmt14.py        dataset download, BPE training, pre-tokenization
  translate.py            translate sentences with a trained checkpoint
  utils/
    parallel_split.py     ParallelSplit: pre-tokenized sentence pairs stored flat with offsets
    batching.py           length-bucketed batches, padding (collate), stripping special tokens
    evaluation.py         translation (shared by training and translate.py), validation loss, BLEU
    checkpoint.py         load_model: rebuild the Transformer from a checkpoint's stored settings
tests/
  gradcheck.py            hand-written backward vs. finite differences / MLX autodiff
```

Shared building blocks (`Module`, `Param`, `Linear`, `Embedding`, `Dropout`, `RMSNorm`, `RoPE`, `SiLU`, `SwiGLUFeedForward`, `MultiHeadAttention`, `EncoderLayer`, `DecoderLayer`, `LabelSmoothingLoss`, `AdamW`, schedules) and training utilities (the background batch `Prefetcher`, `save_checkpoint` / `load_checkpoint`) live in [nnkit](https://github.com/Gevornairi/nnkit).

### How a layer works

Every layer subclasses `Module`. `forward` computes the output and caches what the backward pass needs.
`backward(dy)` takes ∂loss/∂output, adds the parameter gradients into each `Param.grad`, and returns ∂loss/∂input.
A training step is therefore:

```python
model.zero_grad()
logits = model.forward(src_ids, tgt_in_ids)
loss, d_logits = loss_fn(logits, tgt_out_ids)
model.backward(d_logits)
optimizer.step()
```

A parameter used in several places, like the shared embedding table, adds up the gradient from each use.
