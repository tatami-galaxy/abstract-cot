"""Math answer extraction + equivalence grading.

Copied verbatim (extraction + grading portion only) from the self-distill eval
utils so this project has no cross-repo imports. Source: Power-SMC / Hendrycks
MATH grader stack. Public API: `extract_boxed_answer`, `grade_answer`, `is_equiv`.
"""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _remove_boxed(s: str) -> str | None:
    """Strip a leading \\boxed{...} wrapper and return inner content."""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the rightmost non-empty \\boxed{...} or \\fbox{...} answer.

    Searches right-to-left so we skip placeholder boxes like ``\\boxed{{}}``.
    """
    candidates = []
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            candidates.append(idx)
            start = idx + 1

    if not candidates:
        return None

    for idx in sorted(candidates, reverse=True):
        i = idx
        while i < len(text) and text[i] != "{":
            i += 1
        if i >= len(text):
            continue

        right_brace_idx = None
        num_left_braces_open = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                num_left_braces_open += 1
            if text[j] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = j
                    break
            j += 1

        if right_brace_idx is None:
            continue

        retval = text[idx : right_brace_idx + 1]
        content = _remove_boxed(retval) if retval.startswith("\\boxed{") else retval

        if (
            content is not None
            and content.strip().replace("{", "").replace("}", "").strip() != ""
        ):
            return _remove_boxed(retval) if retval.startswith("\\boxed{") else content

    return None


# ---------------------------------------------------------------------------
# Hendrycks MATH normalization
# ---------------------------------------------------------------------------


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
        string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    # Strip a trailing units annotation such as "\\text{ inches}" or
    # "\\mbox{ cm}^2" (with or without the leading space inside the brace).
    for marker in ("\\text{ ", "\\mbox{ ", "\\text{", "\\mbox{"):
        if marker in string:
            return string.split(marker)[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    # Drop ordinal suffixes (e.g. "12^{\\text{th}}" -> "12") before unit
    # stripping, so the "\\text{" inside them isn't mistaken for a unit.
    string = re.sub(r"\^?\{?\\text\{(st|nd|rd|th)\}\}?", "", string)
    # Drop a leading set-membership prefix (e.g. "x \\in [-2,7]" -> "[-2,7]").
    # The negative lookahead keeps this from matching "\\infty".
    string = re.sub(r"^\s*[a-zA-Z]\s*\\in(?![a-zA-Z])\s*", "", string)
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    # Drop a trailing base subscript on base-conversion answers
    # (e.g. "40_9" or "4210_{5}" -> "40" / "4210"). The digit lookbehind keeps
    # this from collapsing variable subscripts like "x_2".
    string = re.sub(r"(?<=[0-9])_\{?[0-9]+\}?$", "", string)
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _normalize_hendrycks(answer: str | None) -> str | None:
    """Hendrycks MATH normalization (math_equivalence)."""
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# ---------------------------------------------------------------------------
# Grader normalization + sympy equivalence (from math_grader.py in Power-SMC)
# ---------------------------------------------------------------------------

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _read_brace_group(s: str, i: int) -> tuple[str | None, int]:
    """Given s[i] == '{', return (inner_content, index_after_matching_'}')."""
    if i >= len(s) or s[i] != "{":
        return None, i
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    return None, i


def _parenthesize_compound_fracs(expr: str) -> str:
    """Rewrite ``\\frac{A}{B}`` as ``((A)/(B))`` when A or B is a compound
    expression (contains +/-), so latex2text doesn't drop the grouping
    (e.g. ``\\frac{11+9a}{20}`` -> ``11+9a/20``). Simple fractions are left
    untouched so they keep their existing normalization path.
    """
    for cmd in ("\\dfrac", "\\tfrac", "\\frac"):
        start = 0
        while True:
            idx = expr.find(cmd, start)
            if idx < 0:
                break
            j = idx + len(cmd)
            while j < len(expr) and expr[j] == " ":
                j += 1
            num, j = _read_brace_group(expr, j)
            if num is None:
                start = idx + len(cmd)
                continue
            while j < len(expr) and expr[j] == " ":
                j += 1
            den, j2 = _read_brace_group(expr, j)
            if den is None:
                start = idx + len(cmd)
                continue
            if any(op in num for op in "+-") or any(op in den for op in "+-"):
                expr = expr[:idx] + f"(({num})/({den}))" + expr[j2:]
                start = idx  # re-scan to catch fracs nested in num/den
            else:
                start = idx + len(cmd)
    return expr


def _parse_latex(expr: str) -> str:
    expr = _parenthesize_compound_fracs(expr)
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize_grader(expr: str) -> str | None:
    """Secondary normalization from the Power-SMC math grader."""
    if expr is None:
        return None

    m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def _count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def _should_allow_eval(expr: str):
    if _count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def _split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


_AFFIRMATIVE = {"yes", "true"}
_NEGATIVE = {"no", "false"}


def _canonical_bool(s: str | None) -> bool | None:
    """Map an affirmative/negative answer to a bool, else None."""
    if s is None:
        return None
    t = re.sub(r"\\(?:text|mathrm|mbox|textbf)\s*\{([^}]*)\}", r"\1", s)
    t = t.strip().strip("$").strip().lower()
    t = t.split(",")[0].strip().rstrip(".")
    if t in _AFFIRMATIVE:
        return True
    if t in _NEGATIVE:
        return False
    return None


def _canonical_infinity(s: str | None) -> str | None:
    """Map a bare (optionally signed) infinity answer to "+inf"/"-inf", else None."""
    if s is None:
        return None
    t = s.strip().strip("$").replace("\\left", "").replace("\\right", "").strip().lower()
    m = re.fullmatch(r"([+-]?)\s*(?:\\infty|\\inf|infinity|inf|oo)", t)
    if m is None:
        return None
    return "-inf" if m.group(1) == "-" else "+inf"


# Tokens that make a scalar sympy comparison unsound: tuples/lists, equations,
# and matrix/transpose-valued answers.
_LATEX_SYMPY_UNSAFE = (
    ",", ";", "=", "^t", "^{t}", "\\top", "\\begin{",
    "pmatrix", "bmatrix", "vmatrix", "\\mathbf", "\\mathbb", "\\det",
)


def _latex_sympy_equiv(pred: str, gold: str) -> bool:
    """Equivalence via a real LaTeX parser + sympy. Used only as an additional
    True-path, so it can never turn a match into a non-match."""
    for s in (pred, gold):
        low = s.lower()
        if any(tok in low for tok in _LATEX_SYMPY_UNSAFE):
            return False
    try:
        from sympy.parsing.latex import parse_latex
    except Exception:
        return False
    try:
        diff = sympy.simplify(parse_latex(gold) - parse_latex(pred))
        return diff == 0
    except Exception:
        return False


def is_equiv(pred: str, gold: str) -> bool:
    """Check answer equivalence."""
    if pred is None:
        return False

    bp, bg = _canonical_bool(pred), _canonical_bool(gold)
    if bp is not None and bg is not None:
        return bp == bg

    ip, ig = _canonical_infinity(pred), _canonical_infinity(gold)
    if ip is not None and ig is not None:
        return ip == ig

    if _is_equiv_core(pred, gold):
        return True

    return _latex_sympy_equiv(pred, gold)


def _is_equiv_core(pred: str, gold: str) -> bool:
    """The two-tier grader (string + sympy normalization)."""
    if pred is None:
        return False

    # Tier 1: Hendrycks normalization
    gold_normalized_h = _normalize_hendrycks(gold)
    pred_normalized_h = _normalize_hendrycks(pred)
    if gold_normalized_h == pred_normalized_h:
        return True

    # Tier 2: Grader normalization + sympy
    gold_normalized = _normalize_grader(gold)
    pred_normalized = _normalize_grader(pred)

    if gold_normalized is None:
        return False
    if gold_normalized == pred_normalized:
        return True
    if len(pred_normalized) == 0:
        return False

    gold_elems = _split_tuple(gold_normalized)
    pred_elems = _split_tuple(pred_normalized)

    if len(gold_elems) > 1 and (
        gold_normalized[0] != pred_normalized[0]
        or gold_normalized[-1] != pred_normalized[-1]
    ):
        return False
    elif len(gold_elems) != len(pred_elems):
        return False
    else:
        for gold_elem, pred_elem in zip(gold_elems, pred_elems):
            if _is_frac(gold_elem) and _is_frac(pred_elem):
                if gold_elem != pred_elem:
                    return False
            elif _str_is_int(gold_elem) != _str_is_int(pred_elem):
                return False
            else:
                if not _are_equal_under_sympy(gold_elem, pred_elem):
                    return False

    return True


# ---------------------------------------------------------------------------
# Multiple-choice handling
# ---------------------------------------------------------------------------

_MC_MARKER = re.compile(r"\(([A-E])\)")


def parse_mc_options(problem: str) -> dict[str, str]:
    """Parse an inline multiple-choice option list into a {letter: value} map."""
    markers = list(_MC_MARKER.finditer(problem))
    if len(markers) < 3:
        return {}
    letters = [m.group(1) for m in markers]
    if letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return {}

    options: dict[str, str] = {}
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(problem)
        value = problem[start:end]
        value = re.sub(r"\\text\{|\\textbf\{|\\mathrm\{|\\mathbf\{", "", value)
        value = value.replace("\\qquad", " ").replace("\\quad", " ")
        value = value.replace("\\,", " ").replace("\\ ", " ")
        value = value.replace("\\(", "").replace("\\)", "").replace("$", "")
        value = value.strip().lstrip("}").strip().strip(".,;:").strip()
        if value and len(value) < 60:
            options[m.group(1)] = value
    return options if len(options) >= 3 else {}


def grade_answer(pred: str | None, gold: str, problem: str = "") -> bool:
    """Equivalence check that also reconciles multiple-choice letters/values."""
    if pred is None:
        return False
    if is_equiv(pred, gold):
        return True

    options = parse_mc_options(problem) if problem else {}
    if options:
        def expand(x: str) -> str:
            return options.get(x.strip().strip("$").strip(), x)

        if is_equiv(expand(pred), expand(gold)):
            return True
    return False
