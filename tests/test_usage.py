"""Token counting is an estimate - these tests pin the ratio, not the accuracy.

The real risk is not being slightly off. It is the CJK case: folding Chinese
into a chars/4 rule undercounts by ~4x, and the image path already prompts in
Chinese, so the estimate would be confidently wrong for a whole class of
requests rather than marginally wrong for all of them.
"""
from __future__ import annotations

import pytest

from usage import anthropic_usage, estimate_tokens, openai_usage, responses_usage


def _wide_count(text):
    """How many characters the estimator should treat as ~1 token each."""
    import unicodedata
    return sum(1 for ch in text if not ch.isascii()
               and (unicodedata.east_asian_width(ch) in ("W", "F")
                    or "CJK" in unicodedata.name(ch, "")))


class TestRatios:
    def test_empty_input(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_english_is_about_four_chars_per_token(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10  # 450 chars
        got = estimate_tokens(text)
        assert 100 <= got <= 120, f"{got} tokens for {len(text)} chars"

    def test_chinese_is_about_one_token_per_character(self):
        """A chars/4 rule would report 4x fewer tokens than this."""
        text = "今天天气很好，我们出去散步吧。" * 5
        got = estimate_tokens(text)
        wide = _wide_count(text)
        narrow = len(text) - wide
        floor = wide + narrow / 4.0
        assert got >= floor * 0.9, (
            f"{got} tokens for {wide} CJK chars - undercounted; a chars/4 ratio "
            "is wrong for CJK"
        )

    def test_japanese_kana(self):
        text = "こんにちは世界、これはテスト。" * 5
        wide = _wide_count(text)
        assert estimate_tokens(text) >= wide * 0.9

    def test_korean(self):
        text = "안녕하세요 세계 이것은 테스트입니다" * 5
        wide = _wide_count(text)
        assert estimate_tokens(text) >= wide * 0.9

    def test_pure_cjk_does_not_fall_back_to_the_ascii_ratio(self):
        """The strongest form of the claim: no narrow characters at all."""
        text = "字" * 200
        assert estimate_tokens(text) >= 180

    def test_mixed_text_lands_between_the_two_ratios(self):
        """Mixed CJK+ASCII must not collapse to either extreme."""
        text = "Fix the bug in 修复登录 in 登录.py" * 10
        cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
        got = estimate_tokens(text)
        floor = cjk + (len(text) - cjk) / 4.0
        assert floor * 0.8 <= got <= floor * 1.2

    def test_monotonic_in_length(self):
        """Counting must never go down as text grows - a client budgeting on it
        would otherwise see a cost go backwards."""
        prev = 0
        for n in range(1, 40):
            got = estimate_tokens("word " * n)
            assert got >= prev, f"{n} words gave {got}, less than the shorter input"
            prev = got

    def test_code_is_counted_at_all(self):
        code = "def f(x):\n    return x + 1\n"
        assert estimate_tokens(code) > 0


class TestUsageShapes:
    def test_openai_shape(self):
        u = openai_usage("hello world", "hi there")
        assert set(u) == {"prompt_tokens", "completion_tokens", "total_tokens"}
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_responses_shape_renames_the_fields(self):
        u = responses_usage("hello world", "hi there")
        assert set(u) == {"input_tokens", "output_tokens", "total_tokens"}
        assert u["total_tokens"] == u["input_tokens"] + u["output_tokens"]

    def test_anthropic_has_no_total(self):
        """Anthropic's object genuinely has no total_tokens field. Adding one
        would be harmless but wrong, and clients may not ignore it."""
        u = anthropic_usage("hello world", "hi there")
        assert set(u) == {"input_tokens", "output_tokens"}
        assert "total_tokens" not in u

    def test_empty_completion_does_not_break_the_total(self):
        """A tool-call turn returns an empty content string; the total must
        still be consistent rather than NaN or missing."""
        u = openai_usage("do a thing", "")
        assert u["completion_tokens"] == 0
        assert u["total_tokens"] == u["prompt_tokens"]

    @pytest.mark.parametrize("fn", [openai_usage, responses_usage, anthropic_usage])
    def test_none_prompt_and_completion(self, fn):
        u = fn(None, None)
        assert u  # a dict, not a crash
        assert all(isinstance(v, int) and v >= 0 for v in u.values())


class TestRealisticSizes:
    def test_a_long_conversation_prompt_is_not_reported_as_free(self):
        """The whole point: a client budget-check must see a non-zero number."""
        transcript = "Human: what does this do?\n\nAssistant: " + "blah " * 4000
        assert estimate_tokens(transcript) > 500

    def test_cjk_prompt_beats_the_ascii_one_of_the_same_length(self):
        ascii_text = "x" * 200
        cjk_text = "字" * 200
        assert estimate_tokens(cjk_text) > estimate_tokens(ascii_text)