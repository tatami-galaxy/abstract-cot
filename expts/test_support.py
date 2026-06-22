"""Fast unit checks for grading, the staged logits processor, and reward parsing.

Run from project root:  python -m expts.test_support
"""

import torch

from .abstract_tokens import BEGIN_ABSTRACT, END_ABSTRACT
from .constraints import (
    AbstractTraceLogitsProcessor,
    ForcedAbstractLogitsProcessor,
    abstract_completion_indices,
    answer_start_index,
    forced_completion_indices,
)
from .grading import extract_boxed_answer, grade_answer
from .reward import answer_segment, correctness_reward, format_reward


def test_grading():
    # DeepMath-style LaTeX answers must grade by math-equivalence, not string eq.
    assert grade_answer("-2/3", "-\\dfrac{2}{3}", "")
    assert grade_answer("\\frac{\\pi}{2}", "\\dfrac{\\pi}{2}", "")
    assert grade_answer("34", "34", "")
    assert grade_answer("\\infty", "\\infty", "")
    assert not grade_answer("35", "34", "")
    assert not grade_answer(None, "34", "")
    assert extract_boxed_answer("blah \\boxed{34} done") == "34"
    assert extract_boxed_answer("no box here") is None
    print("  grading: OK")


def test_layout():
    T = 4
    assert forced_completion_indices(T) == [0, 5]
    assert abstract_completion_indices(T) == [1, 2, 3, 4]
    assert answer_start_index(T) == 6
    print("  layout: OK")


def test_processor():
    V, T = 50, 3
    begin_id, end_id = 40, 41
    abstract_ids = [42, 43, 44, 45]
    proc = AbstractTraceLogitsProcessor(begin_id, end_id, abstract_ids, T)
    prompt_len = 7
    NEG = torch.finfo(torch.float32).min

    # Walk through generation steps, feeding a growing input_ids.
    seq = torch.zeros((2, prompt_len), dtype=torch.long)
    allowed_by_step = {
        0: {begin_id},
        1: set(abstract_ids), 2: set(abstract_ids), 3: set(abstract_ids),
        4: {end_id},
    }
    for step in range(6):
        scores = torch.randn(2, V)
        out = proc(seq, scores)
        if step in allowed_by_step:
            allowed = allowed_by_step[step]
            for v in range(V):
                if v in allowed:
                    assert torch.allclose(out[:, v], scores[:, v]), (step, v)
                else:
                    assert torch.all(out[:, v] == NEG), (step, v)
        else:  # free answer span -> untouched
            assert torch.allclose(out, scores), step
        # append one generated token to simulate decoding progress
        seq = torch.cat([seq, torch.zeros((2, 1), dtype=torch.long)], dim=1)
    print("  processor: OK")


def test_forced_processor():
    V, T = 50, 3
    begin_id, end_id = 40, 41
    # per-row forced traces
    forced = torch.tensor([[42, 43, 44], [45, 44, 42]])
    proc = ForcedAbstractLogitsProcessor(begin_id, end_id, forced)
    prompt_len = 5
    NEG = torch.finfo(torch.float32).min
    seq = torch.zeros((2, prompt_len), dtype=torch.long)
    forced_by_step = {
        0: [begin_id, begin_id],
        1: [42, 45], 2: [43, 44], 3: [44, 42],
        4: [end_id, end_id],
    }
    for step in range(6):
        scores = torch.randn(2, V)
        out = proc(seq, scores)
        if step in forced_by_step:
            for row, fid in enumerate(forced_by_step[step]):
                assert torch.allclose(out[row, fid], scores[row, fid]), (step, row)
                others = [v for v in range(V) if v != fid]
                assert torch.all(out[row, others] == NEG), (step, row)
        else:
            assert torch.allclose(out, scores), step
        seq = torch.cat([seq, torch.zeros((2, 1), dtype=torch.long)], dim=1)
    print("  forced_processor: OK")


def test_format_reward():
    comps = [
        "<begin_abstract><TOKEN_A><end_abstract> ok \\boxed{5}",
        "<begin_abstract><TOKEN_A><end_abstract> rambling no box",
    ]
    assert format_reward(prompts=["p", "p"], completions=comps) == [0.1, 0.0]
    print("  format_reward: OK")


def test_reward():
    comp_correct = f"{BEGIN_ABSTRACT}<TOKEN_A><TOKEN_B>{END_ABSTRACT} thus \\boxed{{34}}"
    comp_wrong = f"{BEGIN_ABSTRACT}<TOKEN_A>{END_ABSTRACT} the answer is \\boxed{{35}}"
    comp_nobox = f"{BEGIN_ABSTRACT}<TOKEN_A>{END_ABSTRACT} i give up"
    # answer_segment must ignore anything before END (e.g. a stray box in trace)
    assert answer_segment(comp_correct).strip().startswith("thus")
    r = correctness_reward(
        prompts=["p", "p", "p"],
        completions=[comp_correct, comp_wrong, comp_nobox],
        final_answer=["34", "34", "34"],
        problem=["", "", ""],
    )
    assert r == [1.0, 0.0, 0.0], r
    print("  reward: OK")


if __name__ == "__main__":
    test_grading()
    test_layout()
    test_processor()
    test_forced_processor()
    test_format_reward()
    test_reward()
    print("ALL OK")
