"""Save and restore model + optimizer state as a single .npz file, or rebuild a model from one."""

import json
from pathlib import Path

import numpy as np
from nnkit.optimisers import AdamW

from ..transformer import Transformer


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


def load_model(path: Path, vocab_size: int) -> Transformer:
    """Build the Transformer described by the checkpoint's training args and load its weights.

    The architecture comes from the checkpoint, not the caller, so it always
    matches the run that produced it. Returns the model in eval mode.
    """
    with np.load(path) as z:
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
    load_checkpoint(path, model)
    model.eval()
    return model
