"""GRPO trainer subclass for cold-start abstract-CoT.

Two overrides on top of TRL's GRPOTrainer (HF generation path, no vLLM):

1. Generation: every `model.generate` call gets the staged
   `AbstractTraceLogitsProcessor` appended, so completions follow the
   BEGIN / abstract-only / END / free layout.

2. Log-probs: `_get_per_token_logps_and_entropies` masks the constrained
   positions to the same support *before* log-softmax, so the policy-gradient
   log-probs are computed under the exact distribution we sampled from
   (renormalized over the abstract vocab). Forced delimiter positions are
   masked to their single token, giving log p = 0 and zero gradient there.

Assumes the DDP / no-shard path: `unwrap_model_for_generation` must return the
same module instance we patch (true for DDP; not for DeepSpeed-3 / FSDP gather).
"""

import torch
from transformers import LogitsProcessorList
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import entropy_from_logits, selective_log_softmax

from .constraints import AbstractTraceLogitsProcessor


class AbstractColdStartGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, begin_id, end_id, abstract_ids, trace_len, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_vllm:
            raise ValueError("AbstractColdStartGRPOTrainer supports the HF generation path only.")

        self.begin_id = begin_id
        self.end_id = end_id
        self.trace_len = trace_len
        # Sorted abstract id tensor for log-prob masking (device set lazily).
        self._abstract_ids = torch.as_tensor(sorted(set(abstract_ids)), dtype=torch.long)
        self._begin_ids = torch.as_tensor([begin_id], dtype=torch.long)
        self._end_ids = torch.as_tensor([end_id], dtype=torch.long)

        self._abstract_processor = AbstractTraceLogitsProcessor(
            begin_id=begin_id, end_id=end_id, abstract_ids=abstract_ids, trace_len=trace_len
        )
        self._install_generation_constraint()

    # ---- override 1: inject the staged logits processor into generation ----
    def _install_generation_constraint(self) -> None:
        proc = self._abstract_processor
        model = self.model  # raw module; unwrap_model_for_generation returns this under DDP
        orig_generate = model.generate
        self._orig_generate = orig_generate  # unpatched generate, for diagnostics

        def generate_with_constraint(*args, **kwargs):
            proc.reset()
            lp = kwargs.pop("logits_processor", None)
            lp = LogitsProcessorList(lp) if lp else LogitsProcessorList()
            lp.append(proc)
            kwargs["logits_processor"] = lp
            return orig_generate(*args, **kwargs)

        model.generate = generate_with_constraint

    # ---- shared: mask logits to the constrained support, position-aware ----
    def _mask_constrained_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, L, V) over completion-relative positions 0..L-1.
        col 0 -> BEGIN only, cols 1..T -> abstract vocab, col T+1 -> END only,
        cols >= T+2 -> unchanged. Out-of-place (autograd-safe)."""
        T, L = self.trace_len, logits.size(1)
        if L < T + 2:
            return logits  # completion shorter than forced layout: shouldn't happen
        neg = torch.finfo(logits.dtype).min
        begin_ids = self._begin_ids.to(logits.device)
        end_ids = self._end_ids.to(logits.device)
        abstract_ids = self._abstract_ids.to(logits.device)

        def keep(seg: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            out = torch.full_like(seg, neg)
            out[..., ids] = seg[..., ids]
            return out

        begin = keep(logits[:, 0:1, :], begin_ids)
        abstract = keep(logits[:, 1 : T + 1, :], abstract_ids)
        end = keep(logits[:, T + 1 : T + 2, :], end_ids)
        answer = logits[:, T + 2 :, :]
        return torch.cat([begin, abstract, end, answer], dim=1)

    # ---- override 2: constrained log-probs (text-only reimplementation) ----
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        **kwargs,  # absorb TRL's compute_aux_loss + multimodal args; cold-start is text-only
    ):
        batch_size = batch_size or input_ids.size(0)
        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]                 # drop next-token pred
            logits = logits[:, -logits_to_keep:, :]    # keep completion positions
            logits.div_(self.temperature)
            logits = self._mask_constrained_logits(logits)  # <-- constrained support

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            all_logps.append(selective_log_softmax(logits, completion_ids))
            if compute_entropy:
                with torch.no_grad():
                    all_entropies.append(entropy_from_logits(logits))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        # TRL >=1.x expects a 3-tuple (logps, entropies, aux_loss). Cold-start uses
        # a dense model, so the MoE load-balancing aux loss is always None.
        return logps, entropies, None
