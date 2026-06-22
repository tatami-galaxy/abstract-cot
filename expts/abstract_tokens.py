"""Shared constants + loader for the abstract-token vocabulary."""

import json
import os

BEGIN_ABSTRACT = "<begin_abstract>"
END_ABSTRACT = "<end_abstract>"


def load_abstract_token_ids(model_dir: str) -> dict:
    """Load the begin/end/abstract token ids saved by vocab_test.py.

    Returns a dict with keys: begin_abstract, end_abstract (ints),
    abstract_token_ids (list[int]), abstract_tokens (list[str]).
    """
    with open(os.path.join(model_dir, "abstract_token_ids.json")) as f:
        return json.load(f)
