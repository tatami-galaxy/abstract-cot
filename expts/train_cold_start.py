"""Cold-start abstract-CoT GRPO baseline.

Forces a fixed-length abstract-token trace, then a free-form answer, rewarded by
binary math correctness. Expected to underperform — the point is to observe the
failure mode (abstract span carries no causal signal) with real diagnostics.

Single GPU smoke test:
    CUDA_VISIBLE_DEVICES=0 python -m expts.train_cold_start --smoke

Multi-GPU DDP via the accelerate configs in config/:
    # 2 GPUs
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file config/ddp_2gpu.yaml \
        -m expts.train_cold_start --model-dir models/<your-model>

    # 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file config/ddp_4gpu.yaml \
        -m expts.train_cold_start --model-dir models/<your-model>
"""

import argparse

from transformers import AutoTokenizer
from trl import GRPOConfig

from .abstract_tokens import load_abstract_token_ids
from .cold_start_grpo import AbstractColdStartGRPOTrainer
from .data import load_deepmath_grpo
from .diagnostics import DiagnosticsCallback
from .reward import correctness_reward, format_reward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="models/Qwen3.5-4B-Base") # local with abstract tokens
    ap.add_argument("--output-dir", default="output/cold_start")
    ap.add_argument("--trace-len", type=int, default=16, help="T: forced abstract tokens per trace")
    ap.add_argument("--answer-budget", type=int, default=512, help="max answer tokens after END")
    ap.add_argument("--num-train", type=int, default=10000)
    ap.add_argument("--num-eval", type=int, default=500)
    ap.add_argument("--num-generations", type=int, default=8, help="GRPO group size G")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--report-to", default="tensorboard")
    ap.add_argument("--diag-steps", type=int, default=50,
                    help="run counterfactual diagnostics every N steps (0 disables)")
    ap.add_argument("--num-probe", type=int, default=128, help="probe-set size for diagnostics")
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end run")
    args = ap.parse_args()

    if args.smoke:
        args.num_train, args.num_eval = 32, 8
        args.per_device_batch, args.num_generations, args.grad_accum = 4, 4, 1
        args.max_steps, args.answer_budget, args.trace_len = 3, 64, 8
        args.diag_steps, args.num_probe = 2, 8

    ids = load_abstract_token_ids(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    train_ds, eval_ds = load_deepmath_grpo(
        tokenizer, num_train=args.num_train, num_eval=args.num_eval
    )

    max_completion_length = args.trace_len + 2 + args.answer_budget

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        temperature=args.temperature,
        beta=0.0,  # no KL / no reference model for the cold-start baseline
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",  # bitsandbytes 8-bit AdamW: ~4x smaller optimizer state
        log_completions=True,
        num_completions_to_print=4,
        report_to=args.report_to,
    )

    trainer = AbstractColdStartGRPOTrainer(
        model=args.model_dir,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        begin_id=ids["begin_abstract"],
        end_id=ids["end_abstract"],
        abstract_ids=ids["abstract_token_ids"],
        trace_len=args.trace_len,
    )

    if args.diag_steps > 0:
        probe = eval_ds.select(range(min(args.num_probe, len(eval_ds))))
        trainer.add_callback(DiagnosticsCallback(
            trainer, probe["prompt"], probe["final_answer"], probe["problem"],
            begin_id=ids["begin_abstract"], end_id=ids["end_abstract"],
            abstract_ids=ids["abstract_token_ids"], trace_len=args.trace_len,
            answer_budget=args.answer_budget, every_steps=args.diag_steps,
        ))

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
