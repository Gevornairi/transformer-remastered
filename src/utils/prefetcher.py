"""Background thread that keeps a queue of ready training batches."""

import queue
import threading

import numpy as np

from .batching import SpecialIds, collate, token_batches
from .parallel_split import ParallelSplit


class Prefetcher:
    """Builds padded numpy batches on a background thread, forever (epoch after epoch)."""

    def __init__(
        self,
        split: ParallelSplit,
        batch_tokens: int,
        special_ids: SpecialIds,
        seed_: int,
        depth: int = 16,
    ) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.epoch = 0
        args = (split, batch_tokens, special_ids, np.random.default_rng(seed_))
        threading.Thread(target=self._run, args=args, daemon=True).start()

    def _run(self, split, batch_tokens, special_ids, rng):
        while True:
            for idx in token_batches(split, batch_tokens, rng):
                self.queue.put((self.epoch, collate(split, idx, *special_ids)))
            self.epoch += 1

    def next(self):
        """Blocks until a batch is ready; returns (epoch, (src, tgt_in, tgt_out))."""
        return self.queue.get()
