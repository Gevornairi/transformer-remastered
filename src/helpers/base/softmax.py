"""Numerically stable softmax, log-softmax, and the softmax backward pass."""

from .backend import Array, xp


def softmax(x: Array, axis: int = -1) -> Array:
    """``exp(x) / sum(exp(x))`` along ``axis``, shifted by the max to avoid overflow."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = xp.exp(shifted)

    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Array, axis: int = -1) -> Array:
    """``log(softmax(x))`` along ``axis``, computed without forming the softmax."""
    shifted = x - x.max(axis=axis, keepdims=True)

    return shifted - xp.log(xp.exp(shifted).sum(axis=axis, keepdims=True))


def softmax_backward(y: Array, dy: Array, axis: int = -1) -> Array:
    """Gradient w.r.t. the softmax input, given its output ``y`` and upstream ``dy``.

    dx = y * (dy - sum(dy * y)), the Jacobian-vector product of
    diag(y) - y y^T without forming the Jacobian.
    """
    return y * (dy - (dy * y).sum(axis=axis, keepdims=True))
