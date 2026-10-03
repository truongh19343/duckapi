"""Session-reuse continuation logic - the source of the 10.3s -> 3s speedup.

This is the most important test file in the suite. The optimisation it protects
silently DROPPED CONTEXT when a client sent a tool result: flatten_conversation()
emits `[tool_result <id>]: ...` lines that are not Human:-labelled, so
_continuation()'s "exactly one ^Human: mark" test matched, it typed only that
Human turn, and the tool output never reached Duck.ai. The model then answered a
question about a tool result it had never seen, with no error anywhere.

`test_tool_result_forces_resend` is written so it fails against the pre-fix code.
"""
from __future__ import annotations

import duckai

PREV = "System: be helpful\n\nHuman: list files\n\nAssistant: ok"


def _session(last_prompt=PREV):
    """A _BrowserSession with no Chrome: only the two attributes
    _continuation() reads are populated."""
    s = duckai._BrowserSession.__new__(duckai._BrowserSession)
    s._last_prompt = last_prompt
    s.page = object()  # truthy; _continuation only checks it is not None
    s.new_chat = False
    return s


class TestPlainContinuation:
    def test_single_follow_up_is_typed_as_delta(self):
        s = _session()
        assert s._continuation(PREV + "\n\nHuman: now read a.py") == "now read a.py"

    def test_multi_word_follow_up_keeps_everything_after_the_label(self):
        s = _session()
        got = s._continuation(PREV + "\n\nHuman: and what about b.py?")
        assert got == "and what about b.py?"

    def test_exact_repeat_yields_nothing(self):
        """Re-sending the identical prompt adds no new turn."""
        s = _session()
        assert s._continuation(PREV) is None


class TestToolResultForcesResend:
    """The regression this whole file exists for."""

    def test_tool_result_forces_resend(self):
        """A tool_result in the delta must NOT be silently dropped.

        flatten_conversation() labels these as `[tool_result <id>]: ...`, not
        `Human: ...`, so the pre-fix "one Human: mark" check happily typed the
        trailing Human turn and threw the tool output away.
        """
        s = _session()
        nxt = PREV + "\n\n[tool_result call_1]: a.py b.py\n\nHuman: now read a.py"
        assert s._continuation(nxt) is None, (
            "tool_result context was dropped: the relay would type only the "
            "final Human turn and Duck.ai would never see the tool output"
        )

    def test_tool_result_with_no_followup_forces_resend(self):
        s = _session()
        assert s._continuation(PREV + "\n\n[tool_result call_1]: a.py") is None

    def test_system_turn_in_delta_forces_resend(self):
        """Anything but an Assistant echo in the delta means full resend."""
        s = _session()
        nxt = PREV + "\n\nSystem: focus on tests\n\nHuman: continue"
        assert s._continuation(nxt) is None

    def test_multiple_human_turns_forces_resend(self):
        s = _session()
        assert s._continuation(PREV + "\n\nHuman: one\n\nHuman: two") is None

    def test_unlabelled_block_in_delta_forces_resend(self):
        """A block with no role label is not an echo; resend."""
        s = _session()
        nxt = PREV + "\n\nsomething unstructured\n\nHuman: continue"
        assert s._continuation(nxt) is None


class TestAssistantEchoIsAllowed:
    """The delta legitimately contains the reply we just read off the page.
    Rejecting it would resend the whole prompt and cost the 3s-vs-10s win the
    README advertises."""

    def test_assistant_echo_then_human_turn_keeps_fast_path(self):
        s = _session()
        nxt = PREV + "\n\nAssistant: ok\n\nHuman: now read a.py"
        assert s._continuation(nxt) == "now read a.py"

    def test_real_client_pattern_keeps_fast_path(self):
        """The exact shape Claude Code sends: system + full history, growing."""
        s = _session()
        nxt = PREV + "\n\nAssistant: ok\n\nHuman: and now b.py"
        assert s._continuation(nxt) == "and now b.py"


class TestNonContinuation:
    def test_no_previous_prompt(self):
        s = _session(last_prompt=None)
        assert s._continuation("Human: fresh start") is None

    def test_no_page(self):
        s = _session()
        s.page = None
        assert s._continuation(PREV + "\n\nHuman: hi") is None

    def test_prompt_does_not_extend_previous(self):
        s = _session()
        assert s._continuation("Human: totally different") is None

    def test_prompt_is_a_strict_prefix_of_previous(self):
        s = _session()
        assert s._continuation(PREV[:20]) is None

    def test_prefix_match_that_is_not_word_aligned_still_rejected(self):
        """startswith alone is not enough - it must extend cleanly."""
        s = _session("Human: hell")
        assert s._continuation("Human: hello there") is None


class TestRoundTripWithFlatten:
    """The two functions only make sense together: main.py produces the
    prompt, duckai.py decides what to type. Guard the seam."""

    def test_real_flattened_conversation_with_tool_result(self):
        import main

        system = "You are terse."
        msgs = [{"role": "user", "content": "list files"}]
        first = main.flatten_conversation(system, msgs)

        msgs2 = [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "a.py b.py", "tool_call_id": "call_1"},
            {"role": "user", "content": "now read a.py"},
        ]
        second = main.flatten_conversation(system, msgs2)

        s = _session(first)
        assert second.startswith(first), "precondition: flatten must be cumulative"
        assert "[tool_result call_1]" in second[len(first):]
        assert s._continuation(second) is None, (
            "the real flatten() output with a tool_result must not take the "
            "delta path"
        )

    def test_real_flattened_conversation_plain_followup(self):
        import main

        system = "You are terse."
        first = main.flatten_conversation(system, [{"role": "user", "content": "list files"}])
        second = main.flatten_conversation(system, [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now read a.py"},
        ])
        s = _session(first)
        assert s._continuation(second) == "now read a.py"
