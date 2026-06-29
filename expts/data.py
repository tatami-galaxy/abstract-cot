"""DeepMath-103K loading + chat-template prompt formatting for cold-start GRPO.

Prompts are rendered with the model tokenizer's chat template (ChatML for Qwen)
as a single user turn, ending at the assistant generation prefix so the forced
abstract trace + answer follow. The user turn carries an explicit \\boxed{}
instruction so the outcome reward can parse a final answer. We keep only
`question` and `final_answer`; `problem` is carried through so the
multiple-choice path of the grader can use it.
"""

from datasets import load_dataset

# Appended to the user turn so the model is asked for a parseable boxed answer.
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def format_prompt_chat(tokenizer, problem: str) -> str:
    """Render `problem` (plus the boxed-answer instruction) as a single user turn
    via the tokenizer's chat template, ending with the assistant generation prefix.

    Returns a string (tokenize=False); TRL leaves non-conversational string prompts
    untouched and the diagnostics path tokenizes the same string, so both decode
    from an identical prompt.
    """
    messages = [{"role": "user", "content": f"{problem}\n\n{BOXED_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_deepmath_grpo(
    tokenizer,
    num_train: int,
    num_eval: int = 500,
    seed: int = 42,
    difficulty: tuple[int, int] | None = None,
):
    """Return (train_ds, eval_ds) with columns: prompt, final_answer, problem.

    `prompt` is chat-template-formatted with `tokenizer`. `difficulty` optionally
    filters to an inclusive [lo, hi] band on the dataset's float `difficulty`
    field before subsetting.
    """
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")

    if difficulty is not None:
        lo, hi = difficulty
        ds = ds.filter(
            lambda x: x.get("difficulty") is not None and lo <= float(x["difficulty"]) <= hi,
            num_proc=4,
        )

    def _map(x):
        return {
            "prompt": format_prompt_chat(tokenizer, x["question"]),
            "final_answer": str(x["final_answer"]),
            "problem": x["question"],
        }

    ds = ds.map(_map, remove_columns=ds.column_names, num_proc=4)
    ds = ds.shuffle(seed=seed)

    eval_ds = ds.select(range(num_eval))
    train_ds = ds.select(range(num_eval, num_eval + num_train))
    return train_ds, eval_ds
