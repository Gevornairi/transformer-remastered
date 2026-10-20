"""Check the hand-written backward pass against an independent gradient.

NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 python -m tests.gradcheck
    central finite differences in float64 (exact to ~1e-6)
python -m tests.gradcheck
    MLX autodiff (mx.grad) of the same forward pass, float32 on the GPU

A tiny model (d_model=8, 2 layers, vocab 7, dropout off) is run on two
hand-made sentence pairs, one of them padded, so masking is exercised too.
Exits non-zero if any configuration fails.
"""

import numpy as np
from nnkit.base import BACKEND, asarray, seed, to_numpy, xp
from nnkit.losses import LabelSmoothingLoss

from src.transformer import Transformer

PAD, BOS, VOCAB_SIZE = 0, 1, 7
SRC = [[3, 4, 5, 6], [5, 3, 0, 0]]
TGT_IN = [[BOS, 6, 5, 4, 3], [BOS, 3, 5, 0, 0]]
TGT_OUT = [[6, 5, 4, 3, 2], [3, 5, 2, 0, 0]]


def check(
    share: bool,
    n_heads: int = 2,
    n_kv_heads: int | None = None,
    logit_cap: float = 0.0,
) -> float:
    """Max gradient error for one configuration, relative to the largest reference gradient."""
    seed(0)
    model = Transformer(
        VOCAB_SIZE,
        VOCAB_SIZE,
        d_model=8,
        n_layers=2,
        n_heads=n_heads,
        d_ff=16,
        dropout=0.0,
        max_len=16,
        share_embeddings=share,
        n_kv_heads=n_kv_heads,
        logit_cap=logit_cap,
    )
    loss_fn = LabelSmoothingLoss(VOCAB_SIZE, PAD, 0.1)
    src, tgt_in, tgt_out = asarray(SRC), asarray(TGT_IN), asarray(TGT_OUT)
    params = model.parameters()

    def loss_of_model():
        return loss_fn(model.forward(src, tgt_in), tgt_out)[0]

    model.zero_grad()
    _, d_logits = loss_fn(model.forward(src, tgt_in), tgt_out)
    model.backward(d_logits)
    manual = [to_numpy(p.grad) for p in params]

    if BACKEND == "mlx":

        def loss_of_values(values):
            for p, v in zip(params, values):
                p.value = v
            return loss_of_model()

        reference = [
            to_numpy(g) for g in xp.grad(loss_of_values)([p.value for p in params])
        ]
    else:
        reference = []
        for p in params:
            num = np.zeros(p.shape)
            for ix in np.ndindex(*p.shape):
                old = p.value[ix]
                p.value[ix] = old + 1e-6
                loss_plus = loss_of_model().item()
                p.value[ix] = old - 1e-6
                loss_minus = loss_of_model().item()
                p.value[ix] = old
                num[ix] = (loss_plus - loss_minus) / 2e-6
            reference.append(num)

    # Error relative to the largest gradient in the model. Per-element relative error
    # is meaningless for gradients that are (nearly) zero, where float32 rounding
    # dominates.
    scale = max(np.abs(r).max() for r in reference)
    return max(float(np.abs(g - r).max()) for g, r in zip(manual, reference)) / scale


def main():
    tol = 1e-4 if BACKEND == "mlx" else 1e-7  # float32 vs float64 rounding
    ok = True
    # (share_embeddings, heads, kv_heads, logit_cap): multi-head, GQA, multi-query,
    # then soft-capping; the tiny model's logits reach ~2.5, so cap 1 saturates
    for share, h, g, cap in [
        (True, 2, None, 0.0),
        (False, 2, None, 0.0),
        (True, 4, 2, 0.0),
        (True, 4, 1, 0.0),
        (True, 2, None, 1.0),
        (False, 2, None, 1.0),
    ]:
        err = check(share, h, g, cap)
        status = "ok" if err < tol else "FAIL"
        ok &= err < tol
        print(
            f"[{BACKEND}] share={share} heads={h} kv_heads={g or h} logit_cap={cap}: "
            f"max err {err:.2e} (tol {tol:.0e})  {status}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
