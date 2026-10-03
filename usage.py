"""Estimated token counts.

**These numbers are estimates, not measurements.** Duck.ai's SSE frames carry no
token counts and there is no public API that does, so nothing exact is available
without shipping a tokenizer — and the obvious one, tiktoken, is trained on
OpenAI's models while the models behind Duck.ai are gpt-oss, gemma, and Claude.
It would return a confident wrong number rather than an honest rough one, and it
adds ~10MB of encoding data for the privilege.

What this buys: a client with a token budget or a cost dashboard sees a plausible
figure instead of a column of zeros. Within roughly 15% is good enough for that.
Anyone billing on this number should bill on something else.

The CJK split matters. BPE vocabularies tokenize Chinese and Japanese close to
one token per character, so folding them into a chars/4 ratio undercounts a
Chinese answer by about 4x. The image path already prompts in Chinese.
"""
from __future__ import annotations

import unicodedata
from typing import Optional

# Average characters per token for non-CJK text. 4 is the usual rule of thumb for
# English prose and code; code is denser, which is why the default errs low on
# long code blocks rather than high.
_CHARS_PER_TOKEN = 4.0


def _is_wide(ch: str) -> bool:
    """True for characters that tokenize at roughly one token each."""
    if not ch.isascii():
        # CJK ideographs, kana, hangul and fullwidth forms. East Asian Width
        # covers the punctuation that also tokenizes densely (、。「」).
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            return True
        return "CJK" in unicodedata.name(ch, "")
    return False


def estimate_tokens(text: Optional[str]) -> int:
    """Approximate the token count of `text`.

    Returns 0 for empty or None input, so callers can pass a reply that turned
    out to be empty without special-casing it.
    """
    if not text:
        return 0
    wide = sum(1 for ch in text if _is_wide(ch))
    narrow = len(text) - wide
    return int(wide + narrow / _CHARS_PER_TOKEN)


def openai_usage(prompt: Optional[str], completion: Optional[str]) -> dict:
    """The OpenAI `usage` object: chat completions and responses."""
    p = estimate_tokens(prompt)
    c = estimate_tokens(completion)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def responses_usage(prompt: Optional[str], completion: Optional[str]) -> dict:
    """The /v1/responses shape, which renames the two fields and keeps total."""
    p = estimate_tokens(prompt)
    c = estimate_tokens(completion)
    return {"input_tokens": p, "output_tokens": c, "total_tokens": p + c}


def anthropic_usage(prompt: Optional[str], completion: Optional[str]) -> dict:
    """The Anthropic `usage` object: no total, and output is reported separately
    because Anthropic clients read it from message_delta during streaming."""
    return {"input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(completion)}