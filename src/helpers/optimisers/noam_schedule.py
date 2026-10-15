"""The original Transformer's learning-rate schedule."""


class NoamSchedule:
    """``factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)``.

    Linear warmup for ``warmup`` steps, then inverse-square-root decay; the
    peak, at ``step == warmup``, is ``factor / sqrt(d_model * warmup)``.
    """

    def __init__(self, d_model: int, warmup: int = 4000, factor: float = 1.0) -> None:
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor

    def __call__(self, step: int) -> float:
        step = max(step, 1)

        return (
            self.factor * self.d_model**-0.5 * min(step**-0.5, step * self.warmup**-1.5)
        )
