"""Base class for every layer: parameter/child bookkeeping, train/eval mode, checkpoints."""

from __future__ import annotations

from .backend import Array, asarray, to_numpy, xp
from .param import Param


class Module:
    """A node in the model tree with its own parameters and child modules.

    There is no autograd: each subclass implements ``forward``, which caches
    whatever its ``backward`` needs, and ``backward``, which takes the gradient
    of the loss w.r.t. the output, adds parameter gradients into ``Param.grad``
    and returns the gradient w.r.t. the input(s). ``forward`` and ``backward``
    signatures vary per layer, so they are not declared here.

    Subclasses must call ``register_params`` / ``register_modules`` for every
    parameter and child; ``parameters``, ``train``/``eval`` and checkpointing
    only see what was registered.
    """

    def __init__(self) -> None:
        self._params: list[Param] = []
        self._children: list[Module] = []
        self.training = True

    def register_params(self, *params: Param) -> None:
        """Make ``params`` part of this module's trainable parameters."""
        self._params.extend(params)

    def register_modules(self, *modules: Module) -> None:
        """Make ``modules`` children of this module."""
        self._children.extend(modules)

    def parameters(self) -> list[Param]:
        """All parameters of this module and its descendants, depth-first.

        Each ``Param`` appears once even if it is shared (e.g. the embedding
        table used by encoder, decoder and output projection). The order is
        deterministic for a given architecture, which checkpoints rely on.
        """
        seen: set[int] = set()
        out: list[Param] = []

        for p in self._params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)

        for child in self._children:
            for p in child.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)

        return out

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zeros."""
        for p in self.parameters():
            p.grad = xp.zeros_like(p.value)

    def train(self, mode: bool = True) -> Module:
        """Set training mode (dropout active) on this module and all descendants."""
        self.training = mode
        for child in self._children:
            child.train(mode)

        return self

    def eval(self) -> Module:
        """Set evaluation mode (dropout off) on this module and all descendants."""
        return self.train(False)

    def state(self) -> dict[str, Array]:
        """Parameter values as NumPy arrays keyed ``p0, p1, ...`` in ``parameters()`` order."""
        return {f"p{i}": to_numpy(p.value) for i, p in enumerate(self.parameters())}

    def load_state(self, state) -> None:
        """Load values saved by ``state``; the architecture must match exactly."""
        params = self.parameters()
        if len(state) != len(params):
            raise ValueError(
                f"checkpoint has {len(state)} params, model has {len(params)}"
            )
        for i, p in enumerate(params):
            v = state[f"p{i}"]
            if tuple(v.shape) != tuple(p.shape):
                raise ValueError(f"p{i}: checkpoint {v.shape} vs model {p.shape}")
            p.value = asarray(v, p.value.dtype)
