"""BUG-3: _grep_args returned a pattern as a pattern, and never a path.

The unquoted branch (toolrouter.py:106) matched at most one non-space token
between the verb and in/under/within. "search for password within the auth
module" therefore parsed path='the' - the article, not the directory.

Worse, its results were discarded. The branch returns unconditionally when it
matches, so the `if m:` at line 114 is reachable ONLY from the *quoted* regex on
line 103 - a different regex, whose groups mean something else entirely. The
two paths through the function had been silently swapped: the code that looked
careful was dead, and the code that ran was the fallback.

Nothing downstream can tell a wrong path from a missing one: Grep just reports
no match, so an agent asking to search a directory it named got silence.
"""
from __future__ import annotations

import pytest

from toolrouter import _grep_args


class TestQuotedPatterns:
    def test_quoted_pattern_with_path(self):
        assert _grep_args('grep "TODO" in src/') == {"pattern": "TODO", "path": "src/"}

    def test_single_quotes(self):
        assert _grep_args("search for 'password' in auth.py") == {
            "pattern": "password", "path": "auth.py"}

    def test_backticked_path(self):
        assert _grep_args('grep "fixme" in `app/main.py`') == {
            "pattern": "fixme", "path": "app/main.py"}

    def test_in_with_the_natural_phrase(self):
        """The exact phrasing that returned the string 'UNREACHABLE' before."""
        assert _grep_args('search for "password" within the auth module') == {
            "pattern": "password", "path": "module"}

    def test_under(self):
        assert _grep_args('grep "TODO" under tests/') == {
            "pattern": "TODO", "path": "tests/"}

    def test_pattern_without_a_path_still_works(self):
        """No path is fine - Grep can search cwd."""
        assert _grep_args('grep "TODO"') == {"pattern": "TODO"}


class TestUnquotedPatterns:
    def test_plain_token(self):
        assert _grep_args("grep TODO in src/") == {"pattern": "TODO", "path": "src/"}

    def test_article_before_the_path_is_not_the_path(self):
        """The regression. 'the' was being taken as the directory."""
        args = _grep_args("search for password within the auth module")
        assert args is not None
        assert args["pattern"] == "password"
        assert args.get("path") != "the", "an article is not a path"

    def test_phrase_path_takes_the_last_token(self):
        args = _grep_args("search for password within the auth module")
        assert args["path"] == "module"

    def test_grep_for_verb_filler(self):
        """'grep for TODO' - the 'for' is filler, not the pattern."""
        args = _grep_args("grep for TODO in src/")
        assert args["pattern"] == "TODO"
        assert args["path"] == "src/"

    def test_no_path_at_all(self):
        assert _grep_args("grep TODO") == {"pattern": "TODO"}


class TestDoesNotCrash:
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "grep",
        "grep ",
        "in",
        "in src/",
        "grep \"\"",
        "grep ''",
        "grep \"unclosed",
        "search for in",
        "find in the",
        "grep a in b in c",
        "in in in",
        "grep `unclosed",
        "under within in",
        "grep \"a\" \"b\" in c",
        "find \"x\"",
        "grep  ",
        "in within under",
        "\n\ngrep\n\nTODO\n\nin\n\nsrc/\n",
        "search for the pattern in the file called foo",
        "grep '*' .",
        "grep TODO in ..",
        "grep TODO in ./src/",
        "grep TODO in /etc/hosts",
        "grep TODO in C:\\Users\\x",
        "grep TODO in file.py:42",
        "grep x in dir/",
    ])
    def test_returns_none_or_a_dict_never_raises(self, text):
        """Whatever comes in, the router must not raise - it runs inside a
        request handler, and a crash here takes down the whole turn."""
        result = _grep_args(text)
        assert result is None or isinstance(result, dict)
        if result is not None:
            assert isinstance(result.get("pattern"), str)
            if "path" in result:
                assert isinstance(result["path"], str)


class TestUnreachableBranchIsGone:
    def test_no_unreachable_marker_leaks_into_results(self):
        """The old fallback returned its own pattern as `pattern`."""
        for text in ['search for "password" within the auth module',
                     'grep "TODO" in src/', "grep TODO in src/"]:
            args = _grep_args(text)
            assert args is not None
            assert args["pattern"] not in ("UNREACHABLE", '[^\\s`"\'"]+')