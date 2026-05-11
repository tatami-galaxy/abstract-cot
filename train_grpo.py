"""
GRPO Training with Optional CoT Masking

Experiment: What happens when we mask the chain-of-thought from the GRPO loss
and only update on the final answer tokens?

Usage:
    # Baseline GRPO (loss on full completion)
    python train_grpo.py --run_name baseline

    # Masked CoT GRPO (loss only on answer tokens after </think>)
    python train_grpo.py --mask_cot --run_name masked_cot
"""

import argparse
import re
import json
import os
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    Looks for patterns like:
      - "#### 42"       (GSM8K gold format)
      - "The answer is 42"
      - "\\boxed{42}"
      - Just the last number in the text as fallback
    """
    # Try #### format
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # Try "the answer is X"
    match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*([+-]?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Try \boxed{X}
    match = re.search(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in the text
    matches = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if matches:
        return matches[-1].replace(",", "")

    return None


def extract_gold_answer(answer_text: str) -> str:
    """Extract numeric answer from GSM8K gold answer string."""
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return answer_text.strip()


def make_reward_fn(gold_answers: dict[str, str]):
    """Create a reward function that checks exact numeric match.

    Args:
        gold_answers: mapping from prompt string to gold answer string
    """
    def reward_fn(completions, prompts, **kwargs):
        rewards = []
        for prompt_list, completion_list in zip(prompts, completions):
            # prompts come as list of message dicts, get the user content
            prompt_text = prompt_list[-1]["content"] if isinstance(prompt_list, list) else prompt_list
            gold = gold_answers.get(prompt_text, None)

            completion_text = completion_list[0]["content"] if isinstance(completion_list, list) else completion_list
            predicted = extract_numeric_answer(completion_text)

            if gold is not None and predicted is not None and predicted.strip() == gold.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Masked CoT GRPO Trainer
# ---------------------------------------------------------------------------

class MaskedCoTGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that masks chain-of-thought tokens from the loss.

    Only tokens after the last </think> tag contribute to the policy gradient.
    The reward/advantage computation is unchanged -- it still uses the full
    completion outcome.
    """

    def _find_think_end_index(self, completion_ids: torch.Tensor) -> int | None:
        """Find the index right after the last </think> in completion token ids.

        Returns the index into completion_ids of the first token AFTER </think>,
        or None if </think> not found.
        """
        if not hasattr(self, "_think_end_ids"):
            self._think_end_ids = self.processing_class.encode(
                "</think>", add_special_tokens=False
            )

        ids = completion_ids.tolist()
        tag_ids = self._think_end_ids
        tag_len = len(tag_ids)

        # Find last occurrence of the </think> token subsequence
        last_match = None
        for i in range(len(ids) - tag_len + 1):
            if ids[i : i + tag_len] == tag_ids:
                last_match = i

        if last_match is None:
            return None

        return last_match + tag_len

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to zero out CoT tokens in the completion mask."""
        # TRL 1.4.0 uses separate prompt_ids/completion_ids, not a single input_ids
        if "completion_mask" in inputs and "completion_ids" in inputs:
            completion_mask = inputs["completion_mask"]
            completion_ids = inputs["completion_ids"]

            for i in range(completion_ids.shape[0]):
                answer_start = self._find_think_end_index(completion_ids[i])

                if answer_start is None:
                    # No </think> found -- zero out entire completion (no gradient)
                    completion_mask[i] = 0
                else:
                    # Zero out CoT tokens, keep only answer tokens after </think>
                    completion_mask[i, :answer_start] = 0

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


# ---------------------------------------------------------------------------
# Callback: monitor CoT statistics
# ---------------------------------------------------------------------------

class CoTMonitorCallback(TrainerCallback):
    """Logs CoT statistics: length, presence of </think>, and example completions."""

    def __init__(self, log_dir: str, tokenizer, log_examples_every: int = 50):
        self.log_dir = log_dir
        self.tokenizer = tokenizer
        self.log_examples_every = log_examples_every
        self.examples_file = os.path.join(log_dir, "cot_examples.jsonl")
        os.makedirs(log_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged. We piggyback to log CoT stats."""
        # The trainer stores recent completions in _completions if available
        model = kwargs.get("model", None)
        if state.global_step % self.log_examples_every != 0:
            return

        # Try to access recent completions from the trainer
        # GRPOTrainer stores them during generation
        processing_class = kwargs.get("processing_class", self.tokenizer)

        # Log whatever metrics we have
        if logs:
            cot_metrics = {k: v for k, v in logs.items() if "cot" in k.lower() or "reward" in k.lower()}
            if cot_metrics:
                with open(os.path.join(self.log_dir, "metrics.jsonl"), "a") as f:
                    entry = {"step": state.global_step, **cot_metrics}
                    f.write(json.dumps(entry) + "\n")


class CompletionLoggerCallback(TrainerCallback):
    """Logs sampled completions at regular intervals to track CoT evolution."""

    def __init__(self, log_dir: str, tokenizer, eval_prompts: list[str], log_every: int = 50):
        self.log_dir = log_dir
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.log_every = log_every
        self.log_file = os.path.join(log_dir, "completion_samples.jsonl")
        os.makedirs(log_dir, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or state.global_step == 0:
            return
        if model is None:
            return

        model.eval()
        samples = []
        for prompt_text in self.eval_prompts[:5]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Think step by step inside <think></think> tags, then give your final answer."},
                {"role": "user", "content": prompt_text},
            ]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                )
            completion = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)

            # Compute CoT stats
            has_think_tags = "</think>" in completion
            think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            cot_text = think_match.group(1) if think_match else ""
            answer_text = completion.split("</think>")[-1] if has_think_tags else completion

            samples.append({
                "step": state.global_step,
                "prompt": prompt_text[:200],
                "completion": completion[:1500],
                "cot_length_chars": len(cot_text),
                "answer_length_chars": len(answer_text),
                "has_think_tags": has_think_tags,
            })

        with open(self.log_file, "a") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        model.train()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the math problem step by step. "
    "Put your reasoning inside <think></think> tags, then provide your final "
    "numeric answer after the tags in the format: #### <number>"
)


def build_dataset(split: str = "train", max_samples: int | None = None):
    """Load GSM8K and format as chat prompts.

    Returns:
        dataset: HF dataset with a "prompt" column (list of message dicts)
        gold_answers: dict mapping user prompt text -> gold numeric answer
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    gold_answers = {
        row["question"]: extract_gold_answer(row["answer"]) for row in ds
    }

    def format_example(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds, gold_answers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO with optional CoT masking")
    parser.add_argument("--mask_cot", action="store_true", help="Mask CoT tokens from loss")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--num_generations", type=int, default=8, help="Completions per prompt")
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--logging_steps", type=int, default=1, help="Log metrics every N steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=50, help="Log completions every N steps")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster generation")
    parser.add_argument("--vllm_gpu_utilization", type=float, default=0.3, help="vLLM GPU memory utilization")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = "masked_cot" if args.mask_cot else "baseline"

    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = os.path.join(args.output_dir, "cot_logs")

    print(f"{'=' * 60}")
    print(f"  GRPO CoT Masking Experiment")
    print(f"  Mode: {'MASKED COT (answer-only loss)' if args.mask_cot else 'BASELINE (full loss)'}")
    print(f"  Model: {args.model_name}")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading GSM8K dataset...")
    train_ds, gold_answers = build_dataset("train", args.max_samples)
    print(f"  Training samples: {len(train_ds)}")

    # Grab a few eval prompts for completion logging
    eval_ds, _ = build_dataset("test", max_samples=10)
    eval_prompts = [ex["prompt"][-1]["content"] for ex in eval_ds]

    # Reward function
    reward_fn = make_reward_fn(gold_answers)

    # Training config
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        report_to="tensorboard",
        log_on_each_node=False,
        # GRPO specific
        beta=0.0,  # KL penalty coefficient
        # vLLM
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_utilization,
    )

    # Choose trainer class
    TrainerClass = MaskedCoTGRPOTrainer if args.mask_cot else GRPOTrainer

    # Callbacks
    callbacks = [
        CoTMonitorCallback(log_dir=log_dir, tokenizer=tokenizer, log_examples_every=args.log_every),
        CompletionLoggerCallback(
            log_dir=log_dir,
            tokenizer=tokenizer,
            eval_prompts=eval_prompts,
            log_every=args.log_every,
        ),
    ]

    # Build trainer
    trainer = TrainerClass(
        model=args.model_name,
        args=training_config,
        train_dataset=train_ds,
        reward_funcs=reward_fn,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"\nTraining complete. Logs saved to {log_dir}")
    print(f"  - completion_samples.jsonl: CoT evolution over training")
    print(f"  - metrics.jsonl: reward and CoT metrics")


if __name__ == "__main__":
    main()
