"""Array backend shared by every layer.

All math goes through ``xp``, which is either ``mlx.core`` or ``numpy``. Both
expose the same NumPy-style API for everything this project uses; the few
operations whose APIs differ (random numbers, scatter-add, forcing lazy
evaluation) are wrapped by the helpers below.

The backend is picked from environment variables at import time, so set them
before anything under ``src/`` is imported:

    TRANSFORMER_BACKEND=mlx      (default) Apple GPU via MLX, float32 only
    TRANSFORMER_BACKEND=numpy    CPU via NumPy
    TRANSFORMER_DTYPE=float64    NumPy only; used for finite-difference gradient checks
"""

import os
from typing import Any

import numpy as np

BACKEND = os.environ.get("TRANSFORMER_BACKEND", "mlx")

xp: Any  # the array module: mlx.core or numpy
if BACKEND == "mlx":
    import mlx.core as xp
elif BACKEND == "numpy":
    xp = np
else:
    raise ValueError(f"unknown TRANSFORMER_BACKEND={BACKEND!r} (use mlx or numpy)")

Array = Any  # an mx.array or np.ndarray

_dtype_name = os.environ.get("TRANSFORMER_DTYPE", "float32")
if BACKEND == "mlx" and _dtype_name != "float32":
    raise ValueError("MLX on GPU only supports TRANSFORMER_DTYPE=float32")
DTYPE = getattr(xp, _dtype_name)  # float dtype of parameters and activations
INT = xp.int32  # dtype of token ids

_np_rng = np.random.default_rng()


def seed(s: int) -> None:
    """Seed the random generator of the active backend."""
    global _np_rng
    _np_rng = np.random.default_rng(s)
    if BACKEND == "mlx":
        xp.random.seed(s)


def asarray(x, dtype=None) -> Array:
    """Convert ``x`` (list, NumPy array, ...) to a backend array, optionally casting."""
    a = xp.array(x)
    return a if dtype is None else a.astype(dtype)


def to_numpy(x: Array) -> np.ndarray:
    """Copy a backend array to a NumPy array (forces evaluation on MLX)."""
    return np.array(x)


def normal(std: float, shape: tuple[int, ...]) -> Array:
    """Samples from N(0, std^2) with the given shape, in ``DTYPE``."""
    if BACKEND == "mlx":
        return (xp.random.normal(shape) * std).astype(DTYPE)
    return _np_rng.normal(0.0, std, shape).astype(DTYPE)


def uniform(low: float, high: float, shape: tuple[int, ...]) -> Array:
    """Samples from U(low, high) with the given shape, in ``DTYPE``."""
    if BACKEND == "mlx":
        return xp.random.uniform(low, high, shape).astype(DTYPE)
    return _np_rng.uniform(low, high, shape).astype(DTYPE)


def keep_mask(p_keep: float, shape: tuple[int, ...]) -> Array:
    """A 0/1 mask in ``DTYPE`` where each entry is 1 with probability ``p_keep``."""
    if BACKEND == "mlx":
        return xp.random.bernoulli(p_keep, shape).astype(DTYPE)
    return (_np_rng.random(shape) < p_keep).astype(DTYPE)


def scatter_add(target: Array, ids: Array, values: Array) -> Array:
    """``target[ids] += values``, accumulating repeated ids.

    Plain ``target[ids] += values`` keeps only one write per repeated id, which
    would drop gradient for tokens that appear more than once in a batch.
    Returns the updated array (a new one on MLX; ``target`` itself on NumPy).
    """
    if BACKEND == "mlx":
        return target.at[ids].add(values)
    np.add.at(target, ids, values)
    return target


def materialize(*trees) -> None:
    """Force MLX's lazy graph to run so it doesn't grow across steps (no-op on NumPy)."""
    if BACKEND == "mlx":
        xp.eval(*trees)
