from __future__ import annotations

"""
Utilities for building BERT-friendly inputs from Question/Answer/Explanation
triples, prioritizing preservation of Question + Answer and truncating
Explanation first to satisfy a max_length constraint.

Primary entrypoint: `encode_qae_expl_truncated_first`.
"""

from typing import List, Sequence

from transformers import PreTrainedTokenizerBase


def build_qae_text(question: str, answer: str, explanation: str) -> str:
    """Compose the canonical Q/A/E string used for embeddings.

    Example output:
        "Question: {question}, Answer: {answer}, Explanation: {explanation}"
    """

    q = (question or "").strip()
    a = (answer or "").strip()
    e = (explanation or "").strip()
    return f"Question: {q}, Answer: {a}, Explanation: {e}"


def encode_qae_expl_truncated_first(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    explanation: str,
    *,
    max_length: int = 512,
    pad_to_max_length: bool = False,
    return_token_type_ids: bool = True,
):
    """Tokenize Q/A/E with Explanation truncated first to fit `max_length`.

    Policy:
    - Keep Question + Answer intact when possible.
    - Truncate Explanation tokens first to satisfy `max_length`.
    - If Q+A alone exceed the budget, truncate Answer (and in the extreme,
      the tail of Question) as a last resort.

    Returns a dict compatible with HF models: {input_ids, attention_mask, (token_type_ids)}.
    """

    # Tokenize segments without special tokens
    q_ids = tokenizer.encode(f"Question: {(question or '').strip()}", add_special_tokens=False)
    a_ids = tokenizer.encode(f", Answer: {(answer or '').strip()}", add_special_tokens=False)
    e_ids = tokenizer.encode(f", Explanation: {(explanation or '').strip()}", add_special_tokens=False)

    # Account for [CLS] and [SEP] (or tokenizer-specific specials)
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length - specials
    if budget < 0:
        budget = 0

    # Try to keep Q + A intact; truncate E first
    if len(q_ids) + len(a_ids) > budget:
        # Not enough room for full Q+A. Degrade gracefully.
        if len(q_ids) > budget:
            q_ids = q_ids[:budget]
            a_ids = []
            e_ids = []
        else:
            # Keep all of Q, truncate A to remaining budget.
            remain = budget - len(q_ids)
            a_ids = a_ids[:max(0, remain)]
            e_ids = []
    else:
        # Room for Q + A; allocate remaining to Explanation
        remain = budget - (len(q_ids) + len(a_ids))
        e_ids = e_ids[:max(0, remain)]

    body_ids = q_ids + a_ids + e_ids

    # Let the tokenizer add specials and build masks
    encoded = tokenizer.prepare_for_model(
        body_ids,
        add_special_tokens=True,
        max_length=max_length,
        padding=("max_length" if pad_to_max_length else False),
        truncation=False,  # we've already controlled truncation
        return_token_type_ids=return_token_type_ids,
    )
    return encoded


def batch_encode_qae_expl_truncated_first(
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[str],
    answers: Sequence[str],
    explanations: Sequence[str],
    *,
    max_length: int = 512,
    pad_to_max_length: bool = False,
    return_token_type_ids: bool = True,
):
    """Vectorized helper over sequences; returns a dict of lists.

    The output keys mirror HF tokenizers' batch outputs.
    """

    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    token_type_ids: List[List[int]] = []

    for q, a, e in zip(questions, answers, explanations):
        out = encode_qae_expl_truncated_first(
            tokenizer,
            q,
            a,
            e,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            return_token_type_ids=return_token_type_ids,
        )
        input_ids.append(out["input_ids"])
        attention_mask.append(out["attention_mask"])
        if return_token_type_ids and "token_type_ids" in out:
            token_type_ids.append(out["token_type_ids"])  # type: ignore[arg-type]

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if return_token_type_ids and token_type_ids:
        batch["token_type_ids"] = token_type_ids
    return batch

