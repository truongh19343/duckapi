"""BUG-4: _ActiveRequestTracker released the slot before the last byte went out.

The class exists for one reason, in its own docstring: _drain_then_exit must not
exit while a StreamingResponse is still going out, or the user gets cut off
mid-sentence. Its `counting_send` then did this on the final body chunk:

    if ... and not message.get("more_body", False):
        released = True
        _active_requests -= 1
    await send(message)          # <-- the write is still in flight here

The decrement runs BEFORE the await. So during the final write - which is
exactly the slow part, and the whole part that matters for a streamed answer -
the counter already reads zero and the slot is free. The mitigation reintroduces
the bug three lines below itself.

Driving raw ASGI, with no TestClient and no httpx: the tracker is a plain ASGI
callable, so scope/receive/send are all the test needs to supply.
"""
from __future__ import annotations

import asyncio

import pytest

import main


@pytest.fixture(autouse=True)
def _reset_counter():
    before = main._active_requests
    yield
    main._active_requests = before


async def _drive(app, *, expect_raise=False):
    """Run one request through the tracker. Returns (count_after).

    Exceptions from the app are captured rather than propagated: a real crash
    is converted to a 500 upstream, and what this test cares about is the
    counter afterwards, not what escaped.
    """
    sent = []

    async def send(message):
        sent.append(message)
        return None

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [], "query_string": b""}
    task = asyncio.create_task(main._ActiveRequestTracker(app)(scope, receive, send))
    if expect_raise:
        with pytest.raises(Exception):
            await asyncio.wait_for(task, timeout=2.0)
    else:
        await asyncio.wait_for(task, timeout=2.0)
    return main._active_requests, sent


def _stream_app(n_chunks):
    """An app that sends a streaming body of n_chunks, like a chat SSE reply."""

    async def app(scope, receive, send):
        for i in range(n_chunks):
            await send({"type": "http.response.body",
                        "body": b"data: chunk\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})

    return app


class TestSlotHeldUntilTheLastWrite:
    async def test_counter_still_held_while_final_body_is_blocked(self):
        """THE regression. While the last write is in flight the count must be 1."""
        released = asyncio.Event()
        let_finish = asyncio.Event()

        async def app(scope, receive, send):
            await send({"type": "http.response.body", "body": b"data: a\n\n",
                        "more_body": True})
            await send({"type": "http.response.body", "body": b"data: b\n\n",
                        "more_body": True})
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})

        observed = {}

        async def send(message):
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                observed["during"] = main._active_requests
                released.set()
                await let_finish.wait()   # the slow write, mid-flight
            return None

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        task = asyncio.create_task(main._ActiveRequestTracker(app)(
            {"type": "http", "method": "POST", "path": "/x", "headers": [],
             "query_string": b""}, receive, send))

        await asyncio.wait_for(released.wait(), timeout=2.0)
        await asyncio.sleep(0)
        assert observed["during"] == 1, (
            "counter released while the final body write is still in flight - "
            "_drain_then_exit would see 0 and exit mid-answer"
        )
        let_finish.set()
        await asyncio.wait_for(task, timeout=2.0)


class TestCounterBalances:
    async def test_returns_to_zero_after_a_normal_stream(self):
        after, _ = await _drive(_stream_app(3))
        assert after == 0

    async def test_balances_when_the_generator_raises_mid_body(self):
        """A crashed stream must not leak a slot; the slot leak would wedge
        shutdown forever, since _drain_then_exit waits on a count that never
        reaches zero."""

        async def app(scope, receive, send):
            await send({"type": "http.response.body", "body": b"data: a\n\n",
                        "more_body": True})
            raise RuntimeError("generator blew up mid-stream")

        after, _ = await _drive(app, expect_raise=True)
        assert after == 0

    async def test_balances_when_the_final_send_raises(self):
        """The transport dying on the last write must not double-decrement."""
        calls = {"n": 0}

        async def app(scope, receive, send):
            await send({"type": "http.response.body", "body": b"data: a\n\n",
                        "more_body": True})
            await send({"type": "http.response.body", "body": b"data: b\n\n"})

        async def send(message):
            calls["n"] += 1
            raise OSError("client hung up")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        with pytest.raises(OSError):
            await asyncio.wait_for(main._ActiveRequestTracker(app)(
                {"type": "http", "method": "POST", "path": "/x", "headers": [],
                 "query_string": b""}, receive, send), timeout=2.0)
        assert main._active_requests == 0

    async def test_balances_on_an_exception_before_any_send(self):
        async def app(scope, receive, send):
            raise RuntimeError("died before writing anything")

        after, _ = await _drive(app, expect_raise=True)
        assert after == 0


class TestNonHttpScopes:
    async def test_websocket_is_not_counted(self):
        """Only HTTP requests should move the counter - a long-lived
        WebSocket would otherwise pin shutdown open indefinitely."""
        ran = []

        async def app(scope, receive, send):
            ran.append(True)

        async def send(message):
            return None

        await main._ActiveRequestTracker(app)(
            {"type": "websocket", "path": "/ws"}, None, send)
        assert ran == [True]
        assert main._active_requests == 0

    async def test_lifespan_is_not_counted(self):
        """Otherwise uvicorn's startup event holds a slot the whole run."""

        async def app(scope, receive, send):
            return None

        async def send(message):
            return None

        await main._ActiveRequestTracker(app)(
            {"type": "lifespan"}, None, send)
        assert main._active_requests == 0


class TestCountingUnderConcurrency:
    async def test_five_concurrent_streams_all_hold_their_slots(self):
        """Each in-flight request must hold exactly one slot - the shutdown
        warning's count is only meaningful if it matches reality."""

        async def send(message):
            return None

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {"type": "http", "method": "POST", "path": "/x", "headers": [],
                 "query_string": b""}
        # Every request parks on the same gate until all five have arrived, so
        # the count is read with all five provably in flight rather than racing.
        # Two events, not one: `all_in` marks arrival, `go` releases them. A
        # single event would let the fifth request run to completion before the
        # test got to read the counter.
        arrived = 0
        all_in = asyncio.Event()
        go = asyncio.Event()

        async def parked_app(scope, receive, send):
            nonlocal arrived
            arrived += 1
            if arrived == 5:
                all_in.set()
            await go.wait()
            await send({"type": "http.response.body", "body": b"x", "more_body": True})
            await send({"type": "http.response.body", "body": b"x"})

        tasks = [asyncio.create_task(
            main._ActiveRequestTracker(parked_app)(scope, receive, send))
            for _ in range(5)]
        await asyncio.wait_for(all_in.wait(), timeout=2.0)
        assert main._active_requests == 5, (
            "in-flight requests must each hold a slot, or the shutdown count "
            "is a fiction"
        )
        go.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert main._active_requests == 0
