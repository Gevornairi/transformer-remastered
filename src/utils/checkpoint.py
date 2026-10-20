"""Rebuild the Transformer from a checkpoint (saving and loading itself is ``nnkit.training``)."""

import json
from pathlib import Path

from nnkit.training import load_checkpoint, read_checkpoint_metadata

from ..transformer import Transformer


def load_model(path: Path, vocab_size: int) -> Transformer:
    """Build the Transformer described by the checkpoint's training args and load its weights.

    The architecture comes from the checkpoint, not the caller, so it always
    matches the run that produced it. Returns the model in eval mode.
    """
    train_args = json.loads(read_checkpoint_metadata(path)["args"])
    model = Transformer(
        vocab_size,
        vocab_size,
        d_model=train_args["d_model"],
        n_layers=train_args["layers"],
        n_heads=train_args["heads"],
        n_kv_heads=train_args["kv_heads"],
        d_ff=train_args["d_ff"],
        dropout=0.0,
        max_len=train_args["max_len"],
        share_embeddings=True,
        # checkpoints from before --logit-cap existed were trained without it
        logit_cap=train_args.get("logit_cap", 0.0),
    )
    load_checkpoint(path, model)
    model.eval()
    return model
