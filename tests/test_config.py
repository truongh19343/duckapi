"""BUG-2: `.env` was silently ignored.

main.py called load_dotenv() on line 53 while importing duckai on line 41.
Python does not defer imports, so duckai.py's module-level os.getenv calls ran
BEFORE .env was loaded. Measured with .env set to non-default values:

    DUCKAI_WARM_MIN=8.8                 read as 2.0
    DUCKAI_WARM_MAX=9.9                 read as 7.0
    DUCKAI_PREWARM=0                    read as True
    DUCKAI_BASE=https://example.invalid read as https://duck.ai

Six variables, no error, no warning. Editing .env changed nothing.

These tests write a real .env, re-import the modules, and assert the values
arrived. They are the reason config.py exists.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _clear_env() -> None:
    """Drop every var config reads, so nothing leaks in from the developer's
    real .env. python-dotenv does NOT overwrite an already-set variable, so a
    leftover would survive a reload and quietly break the test's premise."""
    for k in list(os.environ):
        if k.startswith("DUCKAI_") or k == "PORT":
            os.environ.pop(k, None)


def _restore_modules(saved) -> None:
    for name in ("config", "duckai", "main"):
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.fixture
def reload_all():
    """Import a fresh `config` with the repo's own .env (the normal case).

    Values passed here stand in for a shell export, applied AFTER the clear so
    they win over any .env - which is what python-dotenv intends.
    """

    def _reload(**env):
        _clear_env()
        os.environ.update(env)
        for name in ("config", "duckai", "main"):
            sys.modules.pop(name, None)
        return importlib.import_module("config")

    saved = {n: sys.modules[n] for n in ("config", "duckai", "main") if n in sys.modules}
    yield _reload
    _restore_modules(saved)


@pytest.fixture
def reload_with_env_file():
    """Import a fresh `config` that reads its .env from a directory you choose.

    config resolves .env beside ITSELF (matching run.py's ROOT), not from the
    cwd - so chdir does nothing. The honest way to test a different .env is to
    execute the real config.py from a directory that has one: copy it in,
    import by path, and the module's own ROOT resolves to the temp folder.
    """

    def _reload(dirpath, **env):
        _clear_env()
        os.environ.update(env)
        dst = Path(dirpath) / "config.py"
        shutil.copyfile(ROOT / "config.py", dst)
        sys.modules.pop("config", None)
        spec = importlib.util.spec_from_file_location("config", dst)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    saved = {n: sys.modules[n] for n in ("config", "duckai", "main") if n in sys.modules}
    yield _reload
    _restore_modules(saved)


def _write_env(tmp_path: Path, body: str) -> Path:
    (tmp_path / ".env").write_text(body)
    return tmp_path


class TestDotenvIsHonoured:
    def test_warm_min_arrives(self, tmp_path, reload_with_env_file):
        _write_env(tmp_path, "DUCKAI_WARM_MIN=8.8\n")
        config = reload_with_env_file(tmp_path)
        assert config.WARM_MIN == 8.8

    def test_warm_max_arrives(self, tmp_path, reload_with_env_file):
        _write_env(tmp_path, "DUCKAI_WARM_MAX=9.9\n")
        config = reload_with_env_file(tmp_path)
        assert config.WARM_MAX == 9.9

    def test_prewarm_zero_arrives_as_false(self, tmp_path, reload_with_env_file):
        """Default is on; only an explicit 0/false/no/off turns it off."""
        _write_env(tmp_path, "DUCKAI_PREWARM=0\n")
        config = reload_with_env_file(tmp_path)
        assert config.PREWARM is False

    def test_base_arrives(self, tmp_path, reload_with_env_file):
        _write_env(tmp_path, "DUCKAI_BASE=https://example.invalid\n")
        config = reload_with_env_file(tmp_path)
        assert config.BASE == "https://example.invalid"

    def test_duckai_sees_the_dotenv_values(self, tmp_path, reload_all, reload_with_env_file):
        """The actual bug: duckai's module globals were read pre-dotenv."""
        _write_env(tmp_path, "DUCKAI_WARM_MIN=8.8\nDUCKAI_WARM_MAX=9.9\nDUCKAI_BASE=https://example.invalid\n")
        reload_with_env_file(tmp_path)
        # duckai must import the SAME config, not re-read env itself.
        for name in ("config", "duckai", "main"):
            sys.modules.pop(name, None)
        duckai = importlib.import_module("duckai")
        assert duckai.WARM_MIN == 8.8
        assert duckai.WARM_MAX == 9.9
        assert duckai.BASE == "https://example.invalid"

    def test_main_sees_the_dotenv_values(self, tmp_path, reload_with_env_file):
        _write_env(tmp_path, "DUCKAI_API_KEY=sk-from-env\nDUCKAI_MODEL=model-x\nPORT=9123\n")
        reload_with_env_file(tmp_path)
        for name in ("config", "duckai", "main"):
            sys.modules.pop(name, None)
        main = importlib.import_module("main")
        assert main.API_KEY == "sk-from-env"
        assert main.PORT == 9123
        assert main.DEFAULT == "model-x"


class TestDefaultsWithoutDotenv:
    def test_defaults_when_no_env_file(self, tmp_path, reload_with_env_file):
        config = reload_with_env_file(tmp_path)
        assert config.BASE == "https://duck.ai"
        assert config.WARM_MIN == 2.0
        assert config.WARM_MAX == 7.0
        assert config.PREWARM is True
        assert config.API_KEY == ""
        assert config.PORT == 8080
        assert config.PROXY_POOL == []

    def test_real_environment_wins_over_dotenv(self, tmp_path, reload_with_env_file):
        """A shell export should override the file, as python-dotenv intends."""
        _write_env(tmp_path, "DUCKAI_WARM_MIN=8.8\n")
        config = reload_with_env_file(tmp_path, DUCKAI_WARM_MIN="3.3")
        assert config.WARM_MIN == 3.3


class TestFlagParsing:
    @pytest.mark.parametrize("raw", ["0", "false", "False", "FALSE", "no", "off", ""])
    def test_falsey_values(self, raw, reload_all):
        config = reload_all(DUCKAI_NEW_CHAT=raw)
        assert config.NEW_CHAT is False

    @pytest.mark.parametrize("raw", ["1", "true", "True", "yes", "on", "anything"])
    def test_truthy_values(self, raw, reload_all):
        config = reload_all(DUCKAI_NEW_CHAT=raw)
        assert config.NEW_CHAT is True

    def test_default_is_off(self, reload_all):
        """Tool routing stays opt-in; it is documented as a gamble."""
        config = reload_all()
        assert config.TOOL_ROUTING is False
        assert config.NEW_CHAT is False

    def test_whitespace_is_tolerated(self, reload_all):
        config = reload_all(DUCKAI_NEW_CHAT="  1  ")
        assert config.NEW_CHAT is True


class TestProxyPool:
    def test_comma_separated_pool(self, reload_all):
        config = reload_all(DUCKAI_PROXIES="http://a:1, http://b:2 ,http://c:3")
        assert config.PROXY_POOL == ["http://a:1", "http://b:2", "http://c:3"]

    def test_empty_entries_are_dropped(self, reload_all):
        config = reload_all(DUCKAI_PROXIES="http://a:1,,  ,http://b:2")
        assert config.PROXY_POOL == ["http://a:1", "http://b:2"]

    def test_single_proxy_shorthand(self, reload_all):
        config = reload_all(DUCKAI_PROXY="http://solo:9")
        assert config.PROXY_POOL == ["http://solo:9"]

    def test_pool_wins_over_shorthand(self, reload_all):
        config = reload_all(DUCKAI_PROXIES="http://pool:1", DUCKAI_PROXY="http://single:2")
        assert config.PROXY_POOL == ["http://pool:1"]

    def test_blank_pool_is_empty_not_a_proxy(self, reload_all):
        """A trailing comma must not become a '' entry - that would be handed
        to Playwright as a proxy URL."""
        config = reload_all(DUCKAI_PROXIES=" , , ")
        assert config.PROXY_POOL == []


class TestAsDict:
    def test_as_dict_never_leaks_the_key(self, reload_all):
        config = reload_all(DUCKAI_API_KEY="sk-super-secret")
        d = config.as_dict()
        assert d["api_key_set"] is True
        assert "sk-super-secret" not in repr(d), "the key must never reach /api/status"

    def test_as_dict_reports_defaults(self, reload_all):
        config = reload_all()
        d = config.as_dict()
        assert d["base"] == "https://duck.ai"
        assert d["warm_min"] == 2.0
        assert d["warm_max"] == 7.0
        assert d["proxy_count"] == 0
        assert d["chrome_path_set"] is False