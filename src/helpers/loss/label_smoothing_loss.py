"""Label-smoothed cross-entropy with its gradient w.r.t. the logits."""

from ..base.backend import DTYPE, Array, xp
from ..base.softmax import log_softmax


class LabelSmoothingLoss:
    """Cross-entropy against a smoothed target distribution, averaged over non-PAD tokens.

    The target puts ``1 - smoothing`` on the true token and spreads
    ``smoothing`` evenly over every other token except PAD. PAD target
    positions contribute neither loss nor gradient.

    Not a ``Module``: it has no parameters and returns the gradient directly.
    """

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        # the smoothing mass is spread over every class except the true one and PAD
        self.n_spread = max(vocab_size - 2, 1)
        self._non_pad_cols = (xp.arange(vocab_size) != pad_id).astype(DTYPE)

    def __call__(self, logits: Array, targets: Array) -> tuple[Array, Array]:
        """logits (batch, T, vocab), targets (batch, T) -> (loss, d_logits).

        ``loss`` is a 0-d backend array (call ``.item()`` to read it);
        ``d_logits`` has the shape of ``logits``.
        """
        b, t, v = logits.shape
        flat_logits = logits.reshape(-1, v)
        flat_targets = targets.reshape(-1)
        log_probs = log_softmax(flat_logits, axis=-1)
        probs = xp.exp(log_probs)

        valid = (flat_targets != self.pad_id).astype(DTYPE)[:, None]
        n_tokens = xp.maximum(valid.sum(), 1.0)

        is_target = xp.arange(v)[None, :] == flat_targets[:, None]
        target_dist = xp.where(
            is_target, 1.0 - self.smoothing, self.smoothing / self.n_spread
        )
        target_dist = target_dist * self._non_pad_cols[None, :] * valid

        loss = -(target_dist * log_probs).sum() / n_tokens
        # each valid row of target_dist sums to 1, so d/dlogits = p - target on valid rows
        d_logits = (probs * valid - target_dist) / n_tokens

        return loss, d_logits.reshape(b, t, v).astype(DTYPE)
