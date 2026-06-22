"""Counterfactual-sensitivity + abstract-token usage diagnostics.

Central question for cold start: *does the answer depend on the abstract trace?*
If shuffling or randomizing the abstract span leaves the answer unchanged, the
model is ignoring the trace — the cold-start failure signature.

For each probe prompt we greedily decode three times (greedy so any difference is
attributable to the trace, not sampling noise):
  1. policy trace   — abstract tokens chosen by the current policy
  2. shuffled trace — the policy trace, permuted, then force-decoded
  3. random trace   — a uniform-random trace, force-decoded

Reported metrics (prefix `diag/`):
  acc_{policy,shuffled,random}      — answer accuracy under each condition
  answer_change_{shuffled,random}   — frac of prompts whose answer string changes
  trace_unique_tokens               — distinct abstract ids used across traces
  trace_token_entropy_{nats,frac}   — usage entropy (frac of log|V_abs|)

Low answer_change + acc_policy ≈ acc_random  =>  trace is ignored.

Standalone:
    CUDA_VISIBLE_DEVICES=0 python -m expts.diagnostics \
        --model-dir runs/cold_start --num-probe 128
"""

import argparse
import json
import math
from collections import Counter

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    TrainerCallback,
)

from .abstract_tokens import load_abstract_token_ids
from .constraints import AbstractTraceLogitsProcessor, ForcedAbstractLogitsProcessor
from .data import load_deepmath_grpo
from .grading import extract_boxed_answer, grade_answer
from .reward import answer_segment


@torch.no_grad()
def _generate(generate_fn, tokenizer, prompts, processor, max_new_tokens, device):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
    processor.reset()
    out = generate_fn(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([processor]),
        pad_token_id=tokenizer.pad_token_id,
    )
    return out[:, enc["input_ids"].size(1):]  # completion ids only


@torch.no_grad()
def counterfactual_diagnostics(
    generate_fn,
    tokenizer,
    prompts,
    golds,
    problems,
    *,
    begin_id,
    end_id,
    abstract_ids,
    trace_len,
    answer_budget,
    device,
    seed=0,
):
    # datasets 5.x columns are lazy `Column` objects, not lists; coerce so the
    # tokenizer's str/list[str] type check passes and indexing/zip is plain.
    prompts, golds, problems = list(prompts), list(golds), list(problems)
    g = torch.Generator().manual_seed(seed)
    vocab = torch.as_tensor(sorted(set(abstract_ids)), dtype=torch.long)
    max_new = trace_len + 2 + answer_budget

    def answers_from(comp_ids):
        texts = tokenizer.batch_decode(comp_ids, skip_special_tokens=False)
        return [extract_boxed_answer(answer_segment(t)) for t in texts]

    # 1) policy traces (greedy) + answers
    policy_proc = AbstractTraceLogitsProcessor(begin_id, end_id, abstract_ids, trace_len)
    comp = _generate(generate_fn, tokenizer, prompts, policy_proc, max_new, device)
    traces = comp[:, 1 : trace_len + 1].cpu()  # abstract span = completion idx 1..T
    policy_ans = answers_from(comp)

    # 2) shuffled traces (per-row permutation)
    shuf = torch.stack([traces[i][torch.randperm(trace_len, generator=g)] for i in range(traces.size(0))])
    comp_s = _generate(generate_fn, tokenizer, prompts,
                       ForcedAbstractLogitsProcessor(begin_id, end_id, shuf), max_new, device)
    shuf_ans = answers_from(comp_s)

    # 3) random traces
    rand = vocab[torch.randint(0, len(vocab), (traces.size(0), trace_len), generator=g)]
    comp_r = _generate(generate_fn, tokenizer, prompts,
                       ForcedAbstractLogitsProcessor(begin_id, end_id, rand), max_new, device)
    rand_ans = answers_from(comp_r)

    n = len(prompts)

    def acc(answers):
        return sum(grade_answer(a, gold, prob) for a, gold, prob in zip(answers, golds, problems)) / n

    def change_rate(base, other):
        return sum((a or "") != (b or "") for a, b in zip(base, other)) / n

    counts = Counter(traces.flatten().tolist())
    probs = torch.tensor(list(counts.values()), dtype=torch.float)
    probs = probs / probs.sum()
    ent = float(-(probs * probs.log()).sum())

    return {
        "diag/acc_policy": acc(policy_ans),
        "diag/acc_shuffled": acc(shuf_ans),
        "diag/acc_random": acc(rand_ans),
        "diag/answer_change_shuffled": change_rate(policy_ans, shuf_ans),
        "diag/answer_change_random": change_rate(policy_ans, rand_ans),
        "diag/trace_unique_tokens": len(counts),
        "diag/trace_token_entropy_nats": ent,
        "diag/trace_token_entropy_frac": ent / math.log(len(vocab)),
    }


class DiagnosticsCallback(TrainerCallback):
    """Runs the counterfactual probe every `every_steps` and logs the metrics.
    Main-process only; uses the trainer's unpatched generate so the staged
    policy processor isn't double-applied."""

    def __init__(self, trainer, prompts, golds, problems, *, begin_id, end_id,
                 abstract_ids, trace_len, answer_budget, every_steps=50):
        self.trainer = trainer
        self.prompts, self.golds, self.problems = prompts, golds, problems
        self.begin_id, self.end_id = begin_id, end_id
        self.abstract_ids, self.trace_len = abstract_ids, trace_len
        self.answer_budget, self.every_steps = answer_budget, every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0 or state.global_step == 0:
            return
        if state.global_step % self.every_steps != 0:
            return
        if not self.trainer.accelerator.is_main_process:
            return
        model = self.trainer.model
        was_training = model.training
        model.eval()
        metrics = counterfactual_diagnostics(
            self.trainer._orig_generate, self.trainer.processing_class,
            self.prompts, self.golds, self.problems,
            begin_id=self.begin_id, end_id=self.end_id, abstract_ids=self.abstract_ids,
            trace_len=self.trace_len, answer_budget=self.answer_budget,
            device=self.trainer.accelerator.device,
        )
        if was_training:
            model.train()
        print(f"[diag step {state.global_step}] " + json.dumps(metrics, indent=2))
        self.trainer.log(metrics)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="checkpoint (or base) dir to probe")
    ap.add_argument("--ids-dir", default=None, help="dir holding abstract_token_ids.json (default: model-dir)")
    ap.add_argument("--trace-len", type=int, default=16)
    ap.add_argument("--answer-budget", type=int, default=512)
    ap.add_argument("--num-probe", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = load_abstract_token_ids(args.ids_dir or args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    _, eval_ds = load_deepmath_grpo(num_train=1, num_eval=args.num_probe)
    prompts = eval_ds["prompt"]
    golds = eval_ds["final_answer"]
    problems = eval_ds["problem"]

    metrics = counterfactual_diagnostics(
        model.generate, tokenizer, prompts, golds, problems,
        begin_id=ids["begin_abstract"], end_id=ids["end_abstract"],
        abstract_ids=ids["abstract_token_ids"], trace_len=args.trace_len,
        answer_budget=args.answer_budget, device=device, seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
