#!/bin/bash
# Run both baseline and masked-CoT GRPO experiments
# Adjust --max_samples for quick testing vs full run

set -e

MAX_SAMPLES=${1:-500}  # Pass as arg, default 500 for quick test

echo "============================================"
echo "  Running BASELINE GRPO (full loss on CoT)"
echo "============================================"
python train_grpo.py \
    --run_name baseline \
    --max_samples $MAX_SAMPLES \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 512 \
    --learning_rate 5e-7 \
    --log_every 50

echo ""
echo "============================================"
echo "  Running MASKED COT GRPO (answer-only loss)"
echo "============================================"
python train_grpo.py \
    --mask_cot \
    --run_name masked_cot \
    --max_samples $MAX_SAMPLES \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 512 \
    --learning_rate 5e-7 \
    --log_every 50

echo ""
echo "============================================"
echo "  Analyzing results"
echo "============================================"
BASELINE_DIR=$(ls -d outputs/baseline_* | tail -1)
MASKED_DIR=$(ls -d outputs/masked_cot_* | tail -1)
python analyze_results.py --baseline_dir "$BASELINE_DIR" --masked_dir "$MASKED_DIR"

echo "Done! Check cot_comparison.png and the outputs/ directories."
