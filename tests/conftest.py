"""Shared fixtures.

Two rules the whole suite obeys:
  - no test may launch a real Chrome (nothing here touches Playwright's launch)
  - no test may reach the network (Duck.ai is never contacted)

Anything that would need either is driven through a fake instead - see
`FakePage` below and the note in test_iter_events.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if a test tries to open a socket.

    urllib is the only HTTP client duckai.py uses; patching urlopen is enough
    to turn an accidental live call into an immediate, obvious error.
    """
    import urllib.request

    def _blocked(*a, **kw):
        raise AssertionError("test attempted a network call")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setattr(urllib.request, "Request", _blocked)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear every DUCKAI_*/PORT var so the developer's real .env cannot
    change a test's outcome. Tests that care set what they need themselves."""
    for k in list(os.environ):
        if k.startswith("DUCKAI_") or k == "PORT":
            monkeypatch.delenv(k, raising=False)


class FakePage:
    """Stands in for a Playwright page in _BrowserSession.

    _BrowserSession touches a page through exactly these methods:
      page.evaluate(js, arg)  -> the JS hooks (_RESET_JS / _POLL_JS)
      page.is_closed()        -> liveness check
      page.close()            -> teardown
      page.wait_for_timeout() -> the poll sleep
      page.query_selector()   -> _wait_warm readiness probe
      page.query_selector_all() -> _trigger_send's Ask/Send button scan

    `script` lists ONLY the _POLL_JS results. The first evaluate() call is
    _RESET_JS (arming the capture object) and is answered separately, so tests
    describe the stream alone and never have to count hook invocations.

    Each script entry is either a dict (one _POLL_JS result) or an Exception
    instance to raise, modelling a broken transport.
    """

    def __init__(self, script, closed: bool = False):
        self._script = list(script)
        self._idx = 0
        self._closed = closed
        self.evaluated = []
        self.closed_called = False
        self.selector_hits = 0
        self.reset_calls = 0

    async def evaluate(self, js, arg=None):
        self.evaluated.append((js, arg))
        # _RESET_JS arms the capture object; _POLL_JS reads it back. Both contain
        # `chunks:[]`, but only _RESET_JS writes the space-free `done:false`,
        # and only _POLL_JS declares the `(n) =>` cursor argument.
        if js.lstrip().startswith("(rw) =>"):
            self.reset_calls += 1
            return None
        if self._idx >= len(self._script):
            # Script exhausted: an empty poll keeps the loop spinning, which is
            # what a real page that never answers looks like.
            return {"n": 0, "text": "", "done": False, "err": None}
        item = self._script[self._idx]
        self._idx += 1
        if isinstance(item, Exception):
            raise item
        return item

    def is_closed(self):
        return self._closed

    async def close(self):
        self.closed_called = True
        self._closed = True

    async def wait_for_timeout(self, ms):
        return None

    async def query_selector(self, selector):
        """The readiness probe (_wait_warm) and _trigger_send both look for the
        composer. A non-None result keeps them off the browser APIs."""
        return FakeElement()

    async def goto(self, url, **kw):
        return None

    async def query_selector_all(self, selector):
        """_trigger_send scans buttons for Ask/Send. Returning one means the
        click path is taken and Enter is not needed."""
        return [FakeElement()]


class FakeElement:
    """Minimal stand-in for the composer textarea / send button."""

    async def fill(self, text):
        return None

    async def click(self, force=False):
        return None

    async def press(self, key):
        return None

    async def inner_text(self):
        return "Ask"

    async def get_attribute(self, name):
        return None


def poll(n: int, text: str = "", done: bool = False, err: str | None = None, **extra):
    """Build one _POLL_JS result. `n` is the chunk cursor the script advances."""
    out = {"n": n, "text": text, "done": done, "err": err}
    out.update(extra)
    return out


def sse_line(action: str = "success", message: str = "") -> str:
    """One Duck.ai SSE data line, as _POLL_JS would deliver it mid-stream."""
    import json

    return "data: " + json.dumps({"action": action, "message": message}) + "\n"