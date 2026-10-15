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

All math goes through one array module, so the same code runs on the Apple GPU (MLX) or the CPU (NumPy).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). MLX needs Apple silicon; elsewhere, use the NumPy backend.

```sh
uv sync
```

## Usage

```sh
# 1. Data: download WMT14 en-de, sample 1M training pairs, train a 32k shared BPE, pre-tokenize
uv run python -m src.prepare_wmt14 --n-train 1000000 --vocab-size 32000

# 2. Train (Transformer-base defaults; checkpoints go to checkpoints/wmt14_en_de/)
uv run python -m src.train_wmt
uv run python -m src.train_wmt --resume            # continue from last.npz

# 3. Test-set BLEU on newstest2014 (greedy and beam) with best.npz
uv run python -m src.train_wmt --eval-only
```

`uv run python -m src.train_wmt --help` lists every option. Useful ones:
`--kv-heads` (GQA), `--schedule noam`, `--steps`, `--batch-tokens` / `--accum` (tokens per update), `--max-len` (longest sequence; longer sentences are truncated).
With the cosine schedule, pass the same `--steps` when resuming, since the schedule depends on it.

## Gradient check

```sh
# exact: central finite differences in float64 on the CPU
TRANSFORMER_BACKEND=numpy TRANSFORMER_DTYPE=float64 uv run python -m tests.gradcheck

# MLX autodiff of the same forward pass, float32 on the GPU
uv run python -m tests.gradcheck
```

It checks a tiny model with shared and separate embeddings, and with multi-head, grouped-query and multi-query attention.
It exits non-zero if any gradient is off.

## Backend

Chosen with environment variables before `src` is imported:

| Variable | Values | Default |
|---|---|---|
| `TRANSFORMER_BACKEND` | `mlx`, `numpy` | `mlx` |
| `TRANSFORMER_DTYPE` | `float32`, `float64` (NumPy only) | `float32` |

## Layout

```
src/
  transformer.py          Transformer: encoder + decoder + output projection, greedy/beam decoding
  encoder.py, decoder.py  embedding → stack of layers → final RMSNorm
  train_wmt.py            data loading, batching, training loop, BLEU evaluation, checkpoints
  prepare_wmt14.py        dataset download, BPE training, pre-tokenization
  helpers/
    base/                 backend (MLX/NumPy), Module and Param, masks, softmax
    layers/               Linear, Embedding, Dropout, RMSNorm, RoPE, SiLU, SwiGLU,
                          MultiHeadAttention (with GQA), EncoderLayer, DecoderLayer
    loss/                 label-smoothed cross-entropy
    optimisers/           AdamW, cosine and Noam schedules
tests/
  gradcheck.py            hand-written backward vs. finite differences / MLX autodiff
```

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
