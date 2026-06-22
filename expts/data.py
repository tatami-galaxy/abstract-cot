"""DeepMath-103K loading + zero-shot prompt formatting for cold-start GRPO.

Prompt format matches the self-distill eval `raw` mode (built for base models).
We keep only `question` and `final_answer`; `problem` is carried through so the
multiple-choice path of the grader can use it.
"""

from datasets import load_dataset

# Raw (base-model) zero-shot prompt, copied from the self-distill eval setup.
RAW_PROMPT = "Can you solve the following math problem? "
RAW_COT = " Please reason step by step, and put your final answer within \\boxed{}."


def format_prompt_raw(problem: str) -> str:
    return RAW_PROMPT + problem + RAW_COT


def load_deepmath_grpo(
    num_train: int,
    num_eval: int = 500,
    seed: int = 42,
    difficulty: tuple[int, int] | None = None,
):
    """Return (train_ds, eval_ds) with columns: prompt, final_answer, problem.

    `difficulty` optionally filters to an inclusive [lo, hi] band on the
    dataset's float `difficulty` field before subsetting.
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
            "prompt": format_prompt_raw(x["question"]),
            "final_answer": str(x["final_answer"]),
            "problem": x["question"],
        }

    ds = ds.map(_map, remove_columns=ds.column_names, num_proc=4)
    ds = ds.shuffle(seed=seed)

    eval_ds = ds.select(range(num_eval))
    train_ds = ds.select(range(num_eval, num_eval + num_train))
    return train_ds, eval_ds
