"""Linear warmup followed by cosine decay."""

import math


class CosineSchedule:
    """Learning rate as a function of the step number.

    Rises linearly from 0 to ``max_lr`` over ``warmup`` steps, then follows a
    half cosine down to ``min_lr`` at ``total_steps``, and stays there.
    """

    def __init__(
        self, max_lr: float, warmup: int, total_steps: int, min_lr: float = 0.0
    ) -> None:
        if total_steps <= warmup:
            raise ValueError(f"total_steps={total_steps} must exceed warmup={warmup}")
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup = warmup
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.max_lr * step / self.warmup
        progress = min(1.0, (step - self.warmup) / (self.total_steps - self.warmup))
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )
