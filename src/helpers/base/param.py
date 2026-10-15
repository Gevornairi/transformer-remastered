"""Trainable parameter: a value array plus its accumulated gradient."""

from .backend import DTYPE, Array, xp


class Param:
    """A trainable tensor and the gradient accumulated for it.

    Layers add into ``grad`` during ``backward``, so a parameter used in several
    places (like the shared embedding table) sums the contributions of every
    use. The optimizer reads ``grad`` and replaces ``value``;
    ``Module.zero_grad`` resets ``grad`` to zeros.
    """

    __slots__ = ("grad", "value")

    def __init__(self, value: Array):
        self.value = xp.array(value).astype(DTYPE)
        self.grad = xp.zeros_like(self.value)

    @property
    def shape(self):
        return self.value.shape
