"""A client that hangs up mid-stream must not leave a half-typed page behind.

When a client disconnects, Starlette cancels the SSE generator's task. That
cancellation unwinds through `async with self._lock` and the lock is released -
fine. What it did NOT do was drop the page. So the browser sat there with a
half-finished turn in the composer and `_last_prompt` still pointing at the
previous turn, and the next request saw page_alive=True and appended a delta to
a page Duck.ai was still writing into.

The drop belongs here, not in main.py: this is the one place that holds the
lock, so it is the one place that knows nobody else is driving the browser.
"""
from __future__ import annotations

import asyncio

import duckai
from tests.conftest import FakePage


def _session():
    """A _BrowserSession with no browser and no real polling."""
    s = duckai._BrowserSession.__new__(duckai._BrowserSession)
    s._lock = asyncio.Lock()
    s.page = FakePage(script=[])
    s._drop_calls = 0
    s._last_prompt = "Human: previous turn\n\nAssistant: earlier reply"

    async def drop():
        s._drop_calls += 1
        s.page = None

    s._drop_page = drop

    async def never(prompt, timeout, rewrite):
        # Yield one token, then block the way a real poll loop does. By the time
        # the test cancels, the turn is genuinely half-done.
        yield {"message": "first "}
        await asyncio.sleep(30)

    s._iter_events = never
    return s


async def _drain(session):
    """Consume the stream until the consumer goes away."""
    out = []
    async for tok in session.send_stream_ui("Human: next question", 120.0):
        out.append(tok)
    return out


class TestDisconnectDropsThePage:
    async def test_cancel_during_a_stream_drops_the_page(self):
        session = _session()
        task = asyncio.create_task(_drain(session))
        await asyncio.sleep(0.05)
        assert task.done() is False, "stream ended early; the test proves nothing"

        # The client goes away: Starlette cancels the generator task.
        task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

        assert session._drop_calls == 1, (
            "the page was left mid-turn after the client disconnected"
        )

    async def test_the_lock_is_still_released_after_a_cancel(self):
        """The other half: a cancelled turn must not wedge the browser.

        If the page were dropped while still holding the lock, every later
        request for this session would block forever.
        """
        session = _session()
        task = asyncio.create_task(_drain(session))
        await asyncio.sleep(0.05)
        task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

        assert not session._lock.locked(), (
            "the browser lock is still held after a client disconnect"
        )

    async def test_a_normal_finish_does_not_drop_the_page(self):
        """Dropping on every exit would throw away a perfectly good warm page
        after each turn, which is the whole point of session reuse."""
        session = _session()

        async def completes(prompt, timeout, rewrite):
            yield {"message": "all "}
            yield {"message": "done"}

        session._iter_events = completes
        assert await _drain(session) == ["all ", "done"]
        assert session._drop_calls == 0, "a clean finish must keep the page"


class TestCancellationPropagates:
    async def test_cancellederror_is_not_swallowed_into_a_retry(self):
        """send_stream's rotation loop retries on Exception. CancelledError is a
        BaseException precisely so it is NOT retried - retrying would start a
        second browser turn for a client that no longer exists."""
        pool = duckai.DuckAISession.__new__(duckai.DuckAISession)
        pool.timeout = 120.0
        pool.max_rotations = 2

        calls = {"n": 0}

        async def boom(prompt, timeout, rewrite):
            calls["n"] += 1
            raise asyncio.CancelledError()
            yield ""  # pragma: no cover - makes this an async generator

        class OneSession:
            async def send_stream_ui(self, prompt, timeout, rewrite):
                async for tok in boom(prompt, timeout, rewrite):
                    yield tok

        async def session_for(idx):
            return OneSession()

        pool._session_for = session_for
        pool._next_proxy_index = lambda: 0

        with __import__("contextlib").suppress(asyncio.CancelledError):
            async for _ in pool.send_stream("hi"):
                pass
        assert calls["n"] == 1, (
            f"the cancelled turn was retried {calls['n']} times; a client that "
            "disconnected should get exactly one attempt"
        )