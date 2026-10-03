"""BUG-6: DUCKAI_WARM_MAX was documented, parsed, and never used.

duckai.py read it at module scope (`WARM_MAX = float(os.getenv(...))`) and
_wait_warm's signature was `async def _wait_warm(self, page, budget: float = 7.0)`.
The default was hardcoded. Setting DUCKAI_WARM_MAX=15 changed nothing, and the
README documented it as the readiness deadline.

Same class of bug as the .env one, and it would have survived that fix: making
the read correct makes WARM_MAX start being *read*, still not *used*.

_wait_warm's floor comes from `self._page_opened_at`, so a test drives the budget
by choosing that timestamp rather than by measuring wall-clock readiness - which
would otherwise pass on a page that happened to answer fast.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import duckai
from tests.conftest import FakePage

READY = {"n": 1, "text": "hi", "done": False, "err": None}


def _cfg(warm_max=None, warm_min=None):
    """A config stand-in carrying only what _wait_warm reads.

    Passing one in is the point of DuckAISession(cfg=...): the warmup budget is
    a property of the session's config, not of a module global, so a test can set
    it without monkeypatching duckai and leaking into every other test.
    """
    real = duckai.config
    return SimpleNamespace(
        WARM_MAX=real.WARM_MAX if warm_max is None else warm_max,
        WARM_MIN=real.WARM_MIN if warm_min is None else warm_min,
    )


def _session(page_opened_seconds_ago: float, cfg=None):
    """A _BrowserSession with no browser: _wait_warm only touches `page`."""
    s = duckai._BrowserSession.__new__(duckai._BrowserSession)
    # cfg too: __new__ skips __init__, and _wait_warm reads the warmup budget off
    # self.cfg rather than a module global, so a session built this way needs one.
    s.cfg = cfg or duckai.config
    loop = asyncio.get_event_loop()
    s._page_opened_at = loop.time() - page_opened_seconds_ago
    return s


def _raise_on_probe(page):
    """Replace the readiness probe so it never goes green.

    _wait_warm catches the exception and returns, which is the fallback path -
    so to observe the BUDGET rather than an early return, the probe has to stay
    None (not ready) and let the loop run to its deadline.
    """

    async def probe(selector):
        page.selector_hits += 1
        await page.wait_for_timeout(50)   # yield, so the deadline can advance
        return None

    return probe


class TestWarmMaxDrivesTheBudget:
    async def test_default_budget_comes_from_config(self, monkeypatch):
        """THE regression: the signature hardcoded 7.0, so the knob did nothing.

        WARM_MAX is set to something other than 7.0 first - asserting against the
        shipped default would pass on the old code, since the hardcoded literal
        happened to equal it.

        Read at call time, not bound as a default argument: `budget: float =
        WARM_MAX` is evaluated once when duckai.py is imported, so it would
        capture whatever config held then and be untestable afterwards.
        """
        session = _session(page_opened_seconds_ago=100.0, cfg=_cfg(warm_max=0.0))
        page = FakePage(script=[])
        # No explicit budget: the call must pick up the patched WARM_MAX of 0.
        await asyncio.wait_for(session._wait_warm(page), timeout=2.0)
        assert page.selector_hits == 0, (
            "the default budget ignored WARM_MAX=0 and still polled the page"
        )

    async def test_default_budget_follows_a_raised_warm_max(self, monkeypatch):
        """A raised ceiling must not return early. A page that never goes ready
        runs the loop to its deadline, so the elapsed time is the budget."""
        session = _session(page_opened_seconds_ago=100.0, cfg=_cfg(warm_max=0.3))
        page = FakePage(script=[])
        page.query_selector = _raise_on_probe(page)

        loop = asyncio.get_event_loop()
        started = loop.time()
        await asyncio.wait_for(session._wait_warm(page), timeout=2.0)
        elapsed = loop.time() - started
        assert elapsed >= 0.25, f"gave up after {elapsed:.3f}s, WARM_MAX was 0.3s"

    async def test_zero_warm_max_means_no_wait_at_all(self, monkeypatch):
        """The extreme: a ceiling of zero polls nothing."""
        session = _session(page_opened_seconds_ago=100.0, cfg=_cfg(warm_max=0.0))
        page = FakePage(script=[])
        page.query_selector = _raise_on_probe(page)
        await asyncio.wait_for(session._wait_warm(page), timeout=2.0)
        assert page.selector_hits == 0

    def test_signature_takes_no_hardcoded_budget(self):
        import inspect

        src = inspect.getsource(duckai._BrowserSession._wait_warm)
        assert "= 7.0" not in src, "the literal 7.0 is back in the signature"

    async def test_zero_budget_skips_the_poll_loop(self):
        """deadline = now + 0, so the while never runs and no probe happens."""
        session = _session(page_opened_seconds_ago=100.0)
        page = FakePage(script=[])
        started = time.monotonic()
        await asyncio.wait_for(session._wait_warm(page, budget=0.0), timeout=2.0)
        assert time.monotonic() - started < 0.2
        assert page.selector_hits == 0, "a zero budget must not probe the page"

    async def test_ready_page_returns_on_the_first_probe(self):
        session = _session(page_opened_seconds_ago=100.0)
        page = FakePage(script=[])
        started = time.monotonic()
        await asyncio.wait_for(session._wait_warm(page, budget=5.0), timeout=2.0)
        # query_selector returns a FakeElement, so this exits on the first pass.
        assert time.monotonic() - started < 0.5
        assert page.selector_hits >= 1


class TestWarmMinIsAFloor:
    async def test_floor_holds_a_page_that_is_ready_too_soon(self):
        """WARM_MIN exists because Duck.ai refuses an instant send while its
        challenge JS boots. A page opened a moment ago must NOT be probed yet,
        or the first turn after an open is the one that earns a ban."""
        session = _session(page_opened_seconds_ago=0.0)   # just opened
        page = FakePage(script=[])
        await asyncio.wait_for(session._wait_warm(page, budget=0.05), timeout=2.0)
        assert page.selector_hits == 0, (
            "probed before WARM_MIN elapsed - the floor is not being enforced"
        )

    async def test_probe_runs_once_the_floor_has_passed(self):
        session = _session(page_opened_seconds_ago=duckai.WARM_MIN + 1.0)
        page = FakePage(script=[])
        await asyncio.wait_for(session._wait_warm(page, budget=5.0), timeout=2.0)
        assert page.selector_hits >= 1, (
            "never probed despite WARM_MIN having elapsed"
        )


class TestFloorAndCeilingTogether:
    async def test_budget_cannot_shorten_the_floor(self):
        """budget=0 is 'give up now', not 'send immediately'. The floor still
        applies, which is what keeps a tiny WARM_MAX from self-inflicting a
        ban on every turn."""
        session = _session(page_opened_seconds_ago=0.0)
        page = FakePage(script=[])
        await asyncio.wait_for(session._wait_warm(page, budget=0.0), timeout=2.0)
        assert page.selector_hits == 0


class TestConfigIsInjected:
    """The original .env bug was env read at import in one module and after
    load_dotenv in another. The fix is config.py plus injection, so this pins the
    injection - a future fifth WARM_* read as a bare global would bring the whole
    class of bug back."""

    def test_default_model_resolves_at_call_time(self):
        """Not a default argument. `model=DEFAULT_MODEL` binds at import and
        would capture whatever config held then, which is the same trap the
        warmup budget had."""
        cfg = SimpleNamespace(DEFAULT_MODEL="from-injected-config",
                              PREWARM=False, WARM_MIN=0.0, WARM_MAX=0.0)
        pool = duckai.DuckAISession(cfg=cfg)
        assert pool.model == "from-injected-config"

    def test_model_argument_still_wins(self):
        cfg = SimpleNamespace(DEFAULT_MODEL="ignored", PREWARM=False,
                              WARM_MIN=0.0, WARM_MAX=0.0)
        assert duckai.DuckAISession(model="explicit", cfg=cfg).model == "explicit"

    async def test_prewarm_respects_injected_config(self):
        cfg = SimpleNamespace(DEFAULT_MODEL="m", PREWARM=False,
                              WARM_MIN=0.0, WARM_MAX=0.0)
        pool = duckai.DuckAISession(cfg=cfg)
        # PREWARM off must return without touching a browser at all.
        await pool.prewarm()
        assert pool._sessions == {}
