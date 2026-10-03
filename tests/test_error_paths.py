"""Error paths in _BrowserSession._iter_events.

BUG-3: the two branches above `self._raise_err(...)` called `_drop_page()` but
that one did not. A server-side error (ERR_BN_LIMIT, ERR_CHALLENGE) therefore
left the dead page attached AND `_last_prompt` pointing at the previous turn.
The next request saw `page_alive=True`, took the continuation path, and typed a
delta onto a page Duck.ai had already cut off - a second silent failure stacked
on top of the first one.

These replay a whole SSE exchange through a fake page: no Chrome, no network.

A note on setup, because it is easy to get wrong: _iter_events() reuses the
attached page ONLY when _continuation() returns a delta. That needs
_last_prompt to be a prefix of the prompt AND the remainder to hold exactly one
`^Human: ` mark. PROMPT/_PREV below are built to satisfy both, so these tests
exercise the page-reuse path. A non-prefix (or _last_prompt=None) sends the code
down the fresh-page path instead, which needs a real browser context.
"""
from __future__ import annotations

import asyncio

import pytest

import duckai
from conftest import FakePage, poll, sse_line

# Mirrors what main.flatten_conversation() actually emits: the delta after a
# prior turn is one Assistant echo followed by exactly one new Human turn.
_PREV = "Human: one"
PROMPT = _PREV + "\n\nAssistant: ok\n\nHuman: next"


def _session_with_page(page, last_prompt=_PREV):
    """A _BrowserSession wired to `page`, bypassing launch/warm entirely."""
    s = duckai._BrowserSession.__new__(duckai._BrowserSession)
    s.model = "gpt-5.6-luna"
    s.proxy = None
    s.timeout = 5.0
    s.new_chat = False
    s._pw = None
    s.browser = None
    s.ctx = None
    s.page = page
    s._last_prompt = last_prompt
    s._ready = True
    s.banned = False
    s._page_opened_at = 0.0
    s._lock = asyncio.Lock()
    return s


async def _drain(agen):
    return [ev async for ev in agen]


class TestServerErrorDropsPage:
    """The regression these cover: the page must not survive a server error."""

    async def test_ban_drops_page_and_clears_last_prompt(self):
        page = FakePage([poll(0, err='{"type":"error","ERR_BN_LIMIT"}')])
        s = _session_with_page(page)

        with pytest.raises(duckai.DuckAIRateLimit):
            await _drain(s._iter_events(PROMPT, 5.0, None))

        assert s.page is None, "page survived a ban - next turn would reuse a dead page"
        assert s._last_prompt is None, (
            "_last_prompt survived the error - next turn would compute a delta "
            "against a transcript the page never accepted"
        )
        assert page.closed_called

    async def test_challenge_error_also_drops_page(self):
        page = FakePage([poll(0, err='{"type":"error","ERR_CHALLENGE"}')])
        s = _session_with_page(page)

        with pytest.raises(duckai.DuckAIError):
            await _drain(s._iter_events(PROMPT, 5.0, None))

        assert s.page is None
        assert s._last_prompt is None

    async def test_unknown_server_error_drops_page(self):
        page = FakePage([poll(0, err='{"type":"error","weird"}')])
        s = _session_with_page(page)

        with pytest.raises(duckai.DuckAIError):
            await _drain(s._iter_events(PROMPT, 5.0, None))

        assert s.page is None


class TestReadFailureStillDropsPage:
    """Pre-existing behaviour that must not regress."""

    async def test_pump_read_failure_drops_page(self):
        page = FakePage([poll(0, err="TypeError: fetch failed")])
        s = _session_with_page(page)

        with pytest.raises(duckai.DuckAIError, match="stream read failed"):
            await _drain(s._iter_events(PROMPT, 5.0, None))

        assert s.page is None

    async def test_silent_close_drops_page(self):
        """Duck.ai closing without any reply is not a usable page either."""
        page = FakePage([poll(0, done=True)])
        s = _session_with_page(page)

        with pytest.raises(duckai.DuckAIError, match="without a reply"):
            await _drain(s._iter_events(PROMPT, 5.0, None))

        assert s.page is None

    async def test_timeout_drops_page(self):
        """An exhausted deadline must not leave a half-typed page behind."""
        page = FakePage([])
        s = _session_with_page(page)
        s.timeout = 0.05

        with pytest.raises(duckai.DuckAIError, match="timed out"):
            await _drain(s._iter_events(PROMPT, 0.05, None))

        assert s.page is None


class TestHappyPathRecordsPrompt:
    """The success path must still record the prompt - the whole reuse
    optimisation depends on it, and the fix above touched its neighbour."""

    async def test_success_records_last_prompt_and_keeps_page(self):
        page = FakePage([
            poll(1, sse_line("success", "Hel"), done=False),
            poll(2, sse_line("success", "lo") + "\ndata: [DONE]\n", done=True),
        ])
        s = _session_with_page(page, last_prompt="Human: hi")

        sent = "Human: hi\n\nHuman: second"
        events = await _drain(s._iter_events(sent, 5.0, None))

        tokens = [e.get("message") for e in events if e.get("message")]
        assert tokens == ["Hel", "lo"]
        assert s._last_prompt == sent
        assert s.page is page, "success must NOT drop the page (that is the speedup)"

    async def test_continuation_records_the_full_prompt_not_the_delta(self):
        """_last_prompt stores the WHOLE prompt, so the next turn can diff
        against the full transcript rather than the fragment we typed."""
        page = FakePage([poll(1, sse_line("success", "ok") + "\n", done=True)])
        s = _session_with_page(page, last_prompt="Human: one")

        full = "Human: one\n\nHuman: two"
        await _drain(s._iter_events(full, 5.0, None))

        assert s._last_prompt == full


class TestRewriteRejectedPath:
    async def test_rewrite_rejection_disables_rewrite_globally(self):
        original = duckai._REWRITE_OK
        duckai._REWRITE_OK = True
        try:
            page = FakePage([poll(0, err='{"type":"error","ERR_CHALLENGE"}', rewriteApplied=True)])
            s = _session_with_page(page)

            with pytest.raises(duckai._RewriteRejected):
                await _drain(s._iter_events(PROMPT, 5.0, {"model": "x"}))

            assert duckai._REWRITE_OK is False, "a rejected rewrite must latch off"
            assert s.page is None
        finally:
            duckai._REWRITE_OK = original