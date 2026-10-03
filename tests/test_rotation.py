"""Proxy rotation: three near-identical loops collapse into one.

send(), send_stream() and send_image() each had their own copy of the same
rotation loop. The copies are the problem, not the repetition: the moment one of
them is fixed, the other two silently keep the old behaviour, and there is no
test that notices because there is no single place the behaviour lives.

These tests pin the semantics all three share, BEFORE the collapse, so the
collapse is provably behaviour-preserving.

The semantics, in the order they matter:

1. A DuckAIRateLimit bans that session and rotates. A ban is per-session.
2. DuckAIError propagates immediately - it is not a ban, so rotating would just
   hit the same failure on the next session.
3. Any other exception is a dead Playwright transport: reset the session so the
   next request rebuilds it, and surface as DuckAIError. No rotation, because
   the browser - not the IP - is what failed.
4. Exhausting the rotations raises DuckAIRateLimit.
"""
from __future__ import annotations

import duckai


def _pool(session_count=2):
    """A DuckAISession with fake browsers and no Playwright anywhere."""
    pool = duckai.DuckAISession.__new__(duckai.DuckAISession)
    pool.model = "test-model"
    pool.timeout = 120.0
    pool.max_rotations = 2
    pool.proxies = [None] * session_count
    pool.banned = {i: False for i in range(session_count)}

    class FakeBrowser:
        def __init__(self):
            self.banned = False
            self.resets = 0
            self.closed = False
            # The real _BrowserSession's shape, so debug_state() sees what it
            # sees in production rather than whatever the fake happens to have.
            self._ready = False
            self._page_opened_at = None
            self.new_chat = False
            self.page = None

        async def _reset(self):
            self.resets += 1

        async def close(self):
            self.closed = True

        async def send_ui(self, prompt, timeout, rewrite):
            return f"reply-{self.tag}"

        async def send_image_ui(self, prompt, timeout=None, size=None, rewrite_model=None):
            return {"b64": "AAA", "tag": self.tag}

        async def send_stream_ui(self, prompt, timeout, rewrite):
            for ch in f"reply-{self.tag}":
                yield ch

    browsers = {}
    for i in range(session_count):
        b = FakeBrowser()
        b.tag = i
        browsers[i] = b

    pool._sessions = browsers

    order = []

    def next_index():
        for i in sorted(pool._sessions):
            if not pool._sessions[i].banned:
                order.append(i)
                return i
        return None

    pool._next_proxy_index = next_index

    async def session_for(idx):
        return pool._sessions[idx]

    pool._session_for = session_for
    pool._picks = order
    return pool


class TestNonBanFailuresPropagate:
    async def test_duckai_error_does_not_rotate(self):
        """A DuckAIError is not a ban. Rotating would repeat the same failure on
        the next session and turn one clear error into a confusing one."""
        pool = _pool()

        async def boom(prompt, timeout, rewrite):
            raise duckai.DuckAIError("stream read failed")

        for b in pool._sessions.values():
            b.send_ui = boom

        with pytest_raises(duckai.DuckAIError):
            await pool.send("hi")
        assert len(pool._picks) == 1, "rotated on a non-ban error"

    async def test_transport_failure_resets_and_surfaces(self):
        """Playwright died - that is local, so rebuild and report. Rotating
        would help nothing and hide which session was broken."""
        pool = _pool()

        async def boom(prompt, timeout, rewrite):
            raise OSError("Target page, context or browser has been closed")

        for b in pool._sessions.values():
            b.send_ui = boom

        with pytest_raises(duckai.DuckAIError, match="transport"):
            await pool.send("hi")
        assert pool._sessions[0].resets == 1, "the dead session was not reset"
        assert len(pool._picks) == 1, "a dead browser should not trigger rotation"


class TestBanRotation:
    async def test_a_ban_marks_the_session_and_retries(self):
        pool = _pool()

        async def banned(prompt, timeout, rewrite):
            raise duckai.DuckAIRateLimit("ERR_BN_LIMIT")

        async def ok(prompt, timeout, rewrite):
            return "recovered"

        first = pool._sessions[0]
        first.send_ui = banned
        pool._sessions[1].send_ui = ok

        assert await pool.send("hi") == "recovered"
        assert first.banned is True, "the banned session was not marked"

    async def test_all_sessions_banned_raises_rate_limit(self):
        pool = _pool()

        async def always_banned(prompt, timeout, rewrite):
            raise duckai.DuckAIRateLimit("ERR_BN_LIMIT")

        for b in pool._sessions.values():
            b.send_ui = always_banned

        with pytest_raises(duckai.DuckAIRateLimit):
            await pool.send("hi")


class TestSendImage:
    async def test_image_ban_rotates(self):
        pool = _pool()

        async def banned(prompt, timeout=None, size=None, rewrite_model=None):
            raise duckai.DuckAIRateLimit("ERR_BN_LIMIT")

        pool._sessions[0].send_image_ui = banned
        pool._sessions[1].send_image_ui = lambda *a, **k: _ret({"b64": "BBB"})

        assert await pool.send_image("a cat") == {"b64": "BBB"}
        assert pool._sessions[0].banned is True


class TestSendStream:
    async def test_stream_ban_before_any_token_rotates(self):
        pool = _pool()

        async def banned(prompt, timeout, rewrite):
            raise duckai.DuckAIRateLimit("ERR_BN_LIMIT")
            yield ""  # pragma: no cover - makes it an async generator

        pool._sessions[0].send_stream_ui = banned
        pool._sessions[1].send_stream_ui = lambda p, t, r: _gen("ok")
        # ^ an async generator function, not a lambda returning a coroutine:
        # send_stream iterates the result directly, so a plain lambda here would
        # fail with a TypeError rather than exercising rotation.

        out = [t async for t in pool.send_stream("hi")]
        assert out == ["ok"]


# Small helpers so the fake methods above read as one line each.
async def _ret(value):
    return value


async def _gen(*tokens):
    for t in tokens:
        yield t


def pytest_raises(exc, match=None):
    """contextlib.suppress's raising sibling, so this file needs no pytest
    import at module scope (conftest pulls pytest in anyway)."""
    import pytest
    return pytest.raises(exc, match=match)

class TestDebugState:
    def test_reports_every_session_without_touching_privates(self):
        """The dashboard reads this, not _ready/_page_opened_at directly - a
        rename of any of those broke the operator view at request time."""
        pool = _pool(session_count=3)
        pool._sessions[0]._ready = True
        pool._sessions[1].banned = True
        pool._sessions[2].page = object()

        state = pool.debug_state()
        assert [s["proxy_index"] for s in state] == [0, 1, 2]
        assert state[0]["ready"] is True and state[0]["page_open"] is False
        assert state[1]["banned"] is True
        assert state[2]["page_open"] is True

    def test_page_age_is_none_before_a_page_is_opened(self):
        """_page_opened_at is None, not 0 - round(now - 0) would report the
        loop's uptime as a page age."""
        pool = _pool()
        assert pool.debug_state()[0]["page_age_s"] is None
