"""Every environment variable the relay reads, resolved once.

WHY THIS FILE EXISTS
--------------------
main.py used to call load_dotenv() on line 53 while importing duckai on line 41.
Python does not defer imports, so duckai.py's module-level os.getenv calls ran
BEFORE .env was loaded. Measured with .env set to non-default values:

    DUCKAI_WARM_MIN=8.8    read as 2.0
    DUCKAI_WARM_MAX=9.9    read as 7.0
    DUCKAI_PREWARM=0       read as True
    DUCKAI_BASE=https://example.invalid   read as https://duck.ai

Six variables were silently ignored, with no error: editing .env changed
nothing and nothing said why.

The fix is not "move load_dotenv up one line" - that only works until someone
adds an import and reintroduces it. Reading config through one module that owns
loading makes the ordering impossible to get wrong: import config, then import
anything that needs it.

Config is read once at import, which is also what the code always intended -
none of these are hot-reloadable.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# .env lives beside this file, matching run.py's ROOT. load_dotenv()'s default
# is find_dotenv(), which walks UP from the CALLING FILE's directory looking for
# a .env and stops at the first dir holding setup.py/pyproject.toml - that walks
# out of the project entirely, so it silently found nothing. Naming the path
# explicitly also means uvicorn started from another cwd still reads the right
# file.
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _flag(name: str, default: str = "0") -> bool:
    """True unless explicitly disabled. Accepts 0/false/no/off, any case."""
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


# --- server ---
API_KEY: str = os.getenv("DUCKAI_API_KEY", "").strip()
PORT: int = int(os.getenv("PORT", "8080") or 8080)

# --- duck.ai client ---
BASE: str = os.getenv("DUCKAI_BASE", "https://duck.ai").strip() or "https://duck.ai"
CHROME_PATH: str = os.getenv("DUCKAI_CHROME_PATH", "").strip()

# --- warmup ---
# WARM_MIN is a hard floor on page age (Duck.ai refuses an instant send while its
# challenge JS boots); WARM_MAX caps how long we wait for readiness.
WARM_MIN: float = float(os.getenv("DUCKAI_WARM_MIN", "2.0"))
WARM_MAX: float = float(os.getenv("DUCKAI_WARM_MAX", "7.0"))
PREWARM: bool = _flag("DUCKAI_PREWARM", "1")

# --- routing / behaviour ---
DEFAULT_MODEL: str = os.getenv("DUCKAI_MODEL", "").strip()
NEW_CHAT: bool = _flag("DUCKAI_NEW_CHAT")
TOOL_ROUTING: bool = _flag("DUCKAI_TOOL_ROUTING")

# --- proxies ---
# A comma-separated pool rotates per session; DUCKAI_PROXY is the single-proxy
# shorthand and loses to the pool when both are set.
PROXY_POOL: list[str] = [
    p.strip()
    for p in (
        os.getenv("DUCKAI_PROXIES", "").strip()
        or os.getenv("DUCKAI_PROXY", "").strip()
    ).split(",")
    if p.strip()
]


def as_dict() -> dict:
    """The resolved config, for the dashboard's config display."""
    return {
        "api_key_set": bool(API_KEY),
        "port": PORT,
        "base": BASE,
        "chrome_path_set": bool(CHROME_PATH),
        "warm_min": WARM_MIN,
        "warm_max": WARM_MAX,
        "prewarm": PREWARM,
        "default_model": DEFAULT_MODEL,
        "new_chat": NEW_CHAT,
        "tool_routing": TOOL_ROUTING,
        "proxy_count": len(PROXY_POOL),
    }