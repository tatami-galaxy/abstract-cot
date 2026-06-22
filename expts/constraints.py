"""Staged constrained-decoding for the cold-start abstract-CoT rollout.

Completion layout (deterministic, fixed length T abstract tokens):

    completion idx:  0     1 .. T        T+1        T+2 ..
    token:          BEGIN  z_1 .. z_T    END        answer ... [EOS]
    phase:          forced abstract-only forced     free

The same layout drives three places:
  * generation        -> `AbstractTraceLogitsProcessor`
  * training log-probs -> mask abstract positions to the abstract vocab
  * loss mask          -> drop the 2 forced positions (deterministic, no gradient)
"""

import torch
from transformers import LogitsProcessor


def abstract_completion_indices(trace_len: int) -> list[int]:
    """Completion-relative indices that are abstract tokens (z_1..z_T)."""
    return list(range(1, trace_len + 1))


def forced_completion_indices(trace_len: int) -> list[int]:
    """Completion-relative indices that are forced delimiters (BEGIN, END)."""
    return [0, trace_len + 1]


def answer_start_index(trace_len: int) -> int:
    """First completion-relative index of the free-form answer span."""
    return trace_len + 2


class AbstractTraceLogitsProcessor(LogitsProcessor):
    """Forces BEGIN, constrains the next T tokens to the abstract vocab, forces
    END, then leaves the answer span unconstrained.

    Stateless across vocab but stateful in the generation step counter, so a
    fresh instance (or `reset()`) is required per `generate()` call. Assumes
    synchronous batched decoding with left-padded prompts (all rows share one
    step counter), which is exactly TRL's HF generation path.
    """

    def __init__(self, begin_id: int, end_id: int, abstract_ids: list[int], trace_len: int):
        self.begin_id = begin_id
        self.end_id = end_id
        self.abstract_ids = torch.as_tensor(sorted(set(abstract_ids)), dtype=torch.long)
        self.trace_len = trace_len
        self._prompt_len: int | None = None

    def reset(self) -> None:
        self._prompt_len = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._prompt_len is None:
            self._prompt_len = input_ids.shape[1]
        step = input_ids.shape[1] - self._prompt_len  # idx of the token about to be generated

        if step == 0:
            allowed = torch.tensor([self.begin_id], device=scores.device)
        elif 1 <= step <= self.trace_len:
            allowed = self.abstract_ids.to(scores.device)
        elif step == self.trace_len + 1:
            allowed = torch.tensor([self.end_id], device=scores.device)
        else:
            return scores  # free answer span

        # Keep original scores on the allowed set, push everything else to the
        # dtype min so softmax renormalizes over exactly the allowed support.
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked[:, allowed] = scores[:, allowed]
        return masked


class ForcedAbstractLogitsProcessor(LogitsProcessor):
    """Forces BEGIN, then an *exact* abstract id sequence (per row), then END,
    then leaves the answer free. Used by the counterfactual diagnostic to decode
    an answer conditioned on a chosen (shuffled / random) trace.

    `forced_abstract_ids` is a (B, T) long tensor: row i is the trace forced for
    batch element i. A 1-D (T,) tensor is broadcast to every row.
    """

    def __init__(self, begin_id: int, end_id: int, forced_abstract_ids: torch.Tensor):
        self.begin_id = begin_id
        self.end_id = end_id
        forced = torch.as_tensor(forced_abstract_ids, dtype=torch.long)
        if forced.dim() == 1:
            forced = forced.unsqueeze(0)
        self.forced = forced  # (B, T) or (1, T)
        self.trace_len = self.forced.size(1)
        self._prompt_len: int | None = None

    def reset(self) -> None:
        self._prompt_len = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._prompt_len is None:
            self._prompt_len = input_ids.shape[1]
        step = input_ids.shape[1] - self._prompt_len
        B = scores.size(0)

        if step == 0:
            forced_col = torch.full((B,), self.begin_id, device=scores.device, dtype=torch.long)
        elif 1 <= step <= self.trace_len:
            col = self.forced[:, step - 1].to(scores.device)
            forced_col = col if col.size(0) == B else col.expand(B)  # broadcast 1-row
        elif step == self.trace_len + 1:
            forced_col = torch.full((B,), self.end_id, device=scores.device, dtype=torch.long)
        else:
            return scores

        forced_col = forced_col.unsqueeze(1)  # (B, 1)
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked.scatter_(1, forced_col, scores.gather(1, forced_col))
        return masked


def mask_logits_to_abstract(
    logits: torch.FloatTensor,
    abstract_ids: torch.LongTensor,
) -> torch.FloatTensor:
    """Restrict a (B, V) logits slice to the abstract vocab (for log-prob parity
    with constrained generation). Returns a new tensor."""
    masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
    idx = abstract_ids.to(logits.device)
    masked[:, idx] = logits[:, idx]
    return masked
