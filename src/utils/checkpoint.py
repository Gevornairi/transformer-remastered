"""Save and restore model + optimizer state as a single .npz file."""

import json
from pathlib import Path

import numpy as np

from ..helpers.optimisers.adamw import AdamW
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
