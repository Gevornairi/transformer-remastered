"""AdamW with global-norm gradient clipping."""

from collections.abc import Callable

import numpy as np

from ..base.backend import Array, asarray, materialize, to_numpy, xp
from ..base.param import Param


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        value -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * value)

    where m_hat, v_hat are the bias-corrected moments. Decay is applied only to
    parameters with 2+ dims (weight matrices and the embedding table), not to
    RMSNorm gains. With ``weight_decay=0`` this is plain Adam. ``schedule``
    maps the 1-based step number to the learning rate.

    The defaults (beta2 = 0.98, eps = 1e-9) are the original Transformer's.
    """

    def __init__(
        self,
        params: list[Param],
        schedule: Callable[[int], float],
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-9,
        clip_norm: float | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.schedule = schedule
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.clip_norm = clip_norm
        self.step_count = 0
        self.m = [xp.zeros_like(p.value) for p in self.params]  # first moment
        self.v = [xp.zeros_like(p.value) for p in self.params]  # second moment
        self.last_lr = 0.0
        self.weight_decays = [
            weight_decay if p.value.ndim >= 2 else 0.0 for p in self.params
        ]

    def _clip_grads(self) -> Array:
        """Scale all gradients so their global L2 norm is at most ``clip_norm``.

        Returns the norm before clipping, as a 0-d array.
        """
        total = xp.sqrt(sum((p.grad * p.grad).sum() for p in self.params))
        if self.clip_norm:
            # branch-free so MLX doesn't have to sync to the CPU every step
            scale = xp.minimum(1.0, self.clip_norm / (total + 1e-6))
            for p in self.params:
                p.grad = p.grad * scale

        return total

    def step(self) -> Array:
        """Clip, then update every parameter; returns the pre-clip grad norm as a 0-d array."""
        self.step_count += 1
        grad_norm = self._clip_grads()
        lr = self.schedule(self.step_count)
        self.last_lr = lr
        bias_corr1 = 1.0 - self.beta1**self.step_count
        bias_corr2 = 1.0 - self.beta2**self.step_count
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad * p.grad)
            m_hat = self.m[i] / bias_corr1
            v_hat = self.v[i] / bias_corr2
            p.value = p.value - lr * (
                m_hat / (xp.sqrt(v_hat) + self.eps) + self.weight_decays[i] * p.value
            )

        materialize([p.value for p in self.params], self.m, self.v, grad_norm)

        return grad_norm

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zeros."""
        for p in self.params:
            p.grad = xp.zeros_like(p.value)

    def state(self) -> dict[str, int | np.ndarray]:
        """Step count and both moments, as NumPy arrays keyed ``m0, v0, m1, ...``."""
        out: dict[str, int | np.ndarray] = {"step_count": self.step_count}
        for i in range(len(self.params)):
            out[f"m{i}"] = to_numpy(self.m[i])
            out[f"v{i}"] = to_numpy(self.v[i])
        return out

    def load_state(self, state) -> None:
        """Restore what ``state`` saved; the parameter list must match."""
        self.step_count = int(state["step_count"])
        for i, p in enumerate(self.params):
            self.m[i] = asarray(state[f"m{i}"], p.value.dtype)
            self.v[i] = asarray(state[f"v{i}"], p.value.dtype)
