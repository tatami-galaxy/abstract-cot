"""Binary outcome reward: parse the answer span and grade against final_answer.

The completion is `<begin_abstract> z... <end_abstract> answer...`. Only the
text after the last `<end_abstract>` is the answer; we extract the rightmost
\\boxed{} from it and grade with the math-equivalence grader.
"""

from .abstract_tokens import END_ABSTRACT
from .grading import extract_boxed_answer, grade_answer


def answer_segment(completion: str) -> str:
    """Return the free-form answer span (text after the last END delimiter)."""
    idx = completion.rfind(END_ABSTRACT)
    return completion[idx + len(END_ABSTRACT):] if idx >= 0 else completion


def correctness_reward(prompts, completions, final_answer, problem=None, **kwargs):
    """TRL reward fn. Extra dataset columns (`final_answer`, `problem`) arrive as
    parallel lists. Returns 1.0 for a correct boxed answer, else 0.0.

    Logged as `rewards/correctness_reward/mean` == true accuracy."""
    problems = problem if problem is not None else [""] * len(completions)
    rewards = []
    for comp, gold, prob in zip(completions, final_answer, problems):
        pred = extract_boxed_answer(answer_segment(comp))
        rewards.append(1.0 if grade_answer(pred, gold, prob or "") else 0.0)
    return rewards


def format_reward(prompts, completions, format_bonus=0.1, **kwargs):
    """Small bonus for emitting a parseable \\boxed{} in the answer span.

    Cold-start safety valve: gives within-group reward variance (and thus a
    GRPO gradient) before any answers are correct, pushing the model to at
    least terminate with a parseable answer. Kept separate from correctness
    so true accuracy stays readable in the logs."""
    rewards = []
    for comp in completions:
        has_box = extract_boxed_answer(answer_segment(comp)) is not None
        rewards.append(format_bonus if has_box else 0.0)
    return rewards
