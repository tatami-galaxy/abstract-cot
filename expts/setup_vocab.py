#!/usr/bin/env python
"""Extend a model's vocabulary with Abstract-CoT tokens.

Adds M reserved "abstract" tokens (<TOKEN_A> ... <TOKEN_Z>, <TOKEN_AA>, ...) plus
two delimiters (<begin_abstract>, <end_abstract>) as *normal* (non-special) vocab
tokens that are matched atomically by the tokenizer and never split into BPE
sub-pieces. New embedding rows are initialized to match the statistics of the
pretrained embeddings, then the extended tokenizer + model are saved.

Reference: Ramji, Naseem & Fernandez Astudillo, "Thinking Without Words:
Efficient Latent Reasoning with Abstract Chain-of-Thought" (IBM Research AI).
"""

import argparse

import torch
from tokenizers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer

BEGIN_ABSTRACT = "<begin_abstract>"
END_ABSTRACT = "<end_abstract>"


def abstract_token_names(m: int) -> list[str]:
    """Return m token names using bijective base-26 (spreadsheet column) naming:
    <TOKEN_A>..<TOKEN_Z>, <TOKEN_AA>..<TOKEN_ZZ>, <TOKEN_AAA>, ..."""
    names = []
    for i in range(1, m + 1):  # 1-indexed
        s, n = "", i
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(ord("A") + r) + s
        names.append(f"<TOKEN_{s}>")
    return names


def build_added_tokens(token_strings: list[str]) -> list[AddedToken]:
    """Normal (non-special) tokens, matched atomically and exempt from the
    tokenizer's normalizer so the exact text always maps to a single id."""
    return [
        AddedToken(
            t,
            normalized=False,  # skip the normalizer -> exact atomic match
            special=False,     # ordinary content; not stripped by skip_special_tokens
            single_word=False,
            lstrip=False,
            rstrip=False,
        )
        for t in token_strings
    ]


@torch.no_grad()
def init_new_embeddings(model, new_ids: list[int], old_vocab_size: int, mode: str) -> None:
    """Initialize the newly added embedding rows.

    mode="match": sample N(mu, sigma) per-dim from the existing embeddings
        (random init, but matched in scale so norms stay sane).
    mode="mean": copy the mean of existing embeddings (very stable).
    """
    in_emb = model.get_input_embeddings().weight.data
    tied = getattr(model.config, "tie_word_embeddings", False)
    out_emb = None if tied else model.get_output_embeddings().weight.data

    def fill(emb):
        ref = emb[:old_vocab_size]
        if mode == "mean":
            for t in new_ids:
                emb[t] = ref.mean(0)
        else:  # "match"
            mu, sigma = ref.mean(0), ref.std(0)
            for t in new_ids:
                emb[t] = torch.normal(mu, sigma).to(emb.dtype)

    fill(in_emb)
    if out_emb is not None:
        fill(out_emb)
    print(f"  initialized {len(new_ids)} rows (mode={mode}, tied={tied})")


def verify_atomic(tok, new_tokens: list[str]) -> None:
    """Assert every new token survives tokenization as exactly one id, even when
    placed adjacent without surrounding whitespace."""
    probe = (
        f"x{new_tokens[2]}{new_tokens[3]} "
        f"{BEGIN_ABSTRACT}{new_tokens[-1]}{END_ABSTRACT}y"
    )
    ids = tok(probe, add_special_tokens=False).input_ids
    for t in new_tokens:
        tid = tok.convert_tokens_to_ids(t)
        assert tid != tok.unk_token_id, f"{t} mapped to UNK"
        if t in probe:
            assert ids.count(tid) == probe.count(t), f"{t} was not atomic"
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded.replace(" ", "") == probe.replace(" ", ""), (
        f"round-trip mismatch:\n  in : {probe!r}\n  out: {decoded!r}"
    )
    print("  atomicity + round-trip: OK")


def reload_check(out_dir: str, new_tokens: list[str]) -> None:
    """Reload the saved tokenizer from disk and re-verify that the AddedToken
    config survived serialization (i.e. tokens are still atomic)."""
    print(f"Reload-check from {out_dir} ...")
    reloaded = AutoTokenizer.from_pretrained(out_dir)
    for t in new_tokens:
        assert reloaded.convert_tokens_to_ids(t) != reloaded.unk_token_id, (
            f"{t} missing after reload"
        )
    verify_atomic(reloaded, new_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--num-abstract", type=int, default=64, help="M, size of abstract vocab")
    ap.add_argument("--init", choices=["match", "mean"], default="match")
    ap.add_argument("--pad-to-multiple-of", type=int, default=64)
    ap.add_argument("--out", default=None,
                    help="output dir (default: models/<model_name>, run from project root)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--reload-check", action="store_true",
                    help="after saving, reload the tokenizer from disk and re-verify atomicity")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"models/{args.model.split('/')[-1]}"

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    )
    old_vocab_size = len(tok)

    abstract_tokens = abstract_token_names(args.num_abstract)
    new_tokens = [BEGIN_ABSTRACT, END_ABSTRACT] + abstract_tokens
    print(f"Adding {len(new_tokens)} tokens ({BEGIN_ABSTRACT}, {END_ABSTRACT}, "
          f"{abstract_tokens[0]}..{abstract_tokens[-1]})")

    num_added = tok.add_tokens(build_added_tokens(new_tokens))
    print(f"  tokenizer grew by {num_added}: {old_vocab_size} -> {len(tok)}")

    model.resize_token_embeddings(len(tok), pad_to_multiple_of=args.pad_to_multiple_of)
    print(f"  embedding matrix resized to {model.get_input_embeddings().weight.shape[0]} rows")

    new_ids = tok.convert_tokens_to_ids(new_tokens)
    init_new_embeddings(model, new_ids, old_vocab_size, args.init)

    verify_atomic(tok, new_tokens)

    print(f"Saving to {args.out} ...")
    tok.save_pretrained(args.out)
    model.save_pretrained(args.out)

    # Keep the id list handy for constrained decoding in the RL phase.
    import json
    with open(f"{args.out}/abstract_token_ids.json", "w") as f:
        json.dump(
            {
                "begin_abstract": tok.convert_tokens_to_ids(BEGIN_ABSTRACT),
                "end_abstract": tok.convert_tokens_to_ids(END_ABSTRACT),
                "abstract_token_ids": tok.convert_tokens_to_ids(abstract_tokens),
                "abstract_tokens": abstract_tokens,
            },
            f,
            indent=2,
        )

    if args.reload_check:
        reload_check(args.out, new_tokens)

    print("Done.")


if __name__ == "__main__":
    main()
