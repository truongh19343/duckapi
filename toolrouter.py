"""Relay-side tool routing for Duck.ai (which has NO native tool-calling).

WHY THIS EXISTS
---------------
Duck.ai's web models (GPT/Claude free tier) never emit tool-call markup, no matter
how the prompt is phrased (verified across 10 live requests: 3 formats x 3 models).
This module bridges that gap WITHOUT executing tools on the server (the agent
already has filesystem/shell access and sandboxing):

  Agent --tools=[Read,Bash,...]--> relay detects intent --> returns tool_use block
  Agent executes locally, sends tool_result back
  relay injects tool_result as context --> Duck.ai answers grounded

The relay only SYNTHESIZES the tool_use block from the user's request; it never
runs the tool. This is the safe, agent-compatible shape.

INTENT PARSING
--------------
Agents (Claude Code / Codex / PI) phrase requests very literally ("Read the file
X", "run `cmd`", "find files matching *.go"). We match the latest user turn
against each registered tool's intent patterns and extract arguments. If nothing
matches, we return None and the request flows to Duck.ai as a normal chat.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

# Tools this relay knows how to route. Names match Claude Code's built-ins so
# agents recognise them. Extend here to add coverage.
KNOWN_TOOLS = {
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
}


@dataclass
class RoutedToolCall:
    name: str
    input: dict


# --- intent patterns (latest user turn -> (tool_name, arg_extractor)) ---

def _read_args(text: str) -> Optional[dict]:
    # "read README.md" / "show me the file /path/x" / "cat src/main.go"
    m = re.search(r'\b(?:read|cat|show|open|display|print|view)\b\s+(?:the\s+|me\s+)?(?:file\s+)?[`"\'"]?([^\s`"\'"]+\.\w+|/[^`"\'"\s]+)[`"\'"]?', text, re.IGNORECASE)
    if m:
        return {"file_path": m.group(1)}
    return None


def _write_args(text: str) -> Optional[dict]:
    # "write 'content' to path" / "create file path with ..."
    m = re.search(r'\bwrite\b[^`"\'"]*[`"\'"]?([^\s`"\'"]+\.\w+|/[^`"\'"\s]+)[`"\'"]?', text, re.IGNORECASE)
    if m:
        return {"file_path": m.group(1), "content": ""}
    return None
def _edit_args(text: str) -> Optional[dict]:
    m = re.search(r'\bedit\b[^`"\'"]*[`"\'"]?([^\s`"\'"]+\.\w+|/[^`"\'"\s]+)[`"\'"]?', text, re.IGNORECASE)
    if m:
        return {"file_path": m.group(1), "old_string": "", "new_string": ""}
    return None


def _bash_args(text: str) -> Optional[dict]:
    # fenced shell block: ```sh / ```bash ... ```
    m = re.search(r'```(?:sh|bash|shell|zsh|cmd)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if m:
        return {"command": m.group(1).strip()}
    # "run: cmd" / "execute cmd" / "run the command cmd" -> explicit prefix, take as-is
    m = re.search(r'\b(?:run|execute)\b\s*(?:the\s+)?(?:command\s+)?[:`"\'"]?\s*([^`"\'"\n]{2,300})', text, re.IGNORECASE)
    if m:
        cmd = m.group(1).strip().strip('`"\'"')
        if cmd:
            return {"command": cmd}
    return None




def _glob_args(text: str) -> Optional[dict]:
    # "find files matching **/*.py" / "glob **/*.py" / "list all *.md"
    # grab the first whitespace-delimited token after the keyword that looks
    # like a glob (contains * or / or is a *.ext pattern).
    m = re.search(
        r'\b(?:glob|find\s+files?\s+matching|list\s+(?:all\s+)?files?\s+matching)\b\s+(\S*[\*\/]\S*|\*\S*|\S+\.\w+)',
        text, re.IGNORECASE,
    )
    if m:
        return {"pattern": m.group(1)}
    return None

_GREP_VERB = r'\b(?:grep|search\s+for(?:\s+the)?|find)(?:\s+for)?\b'
# A clause ends at sentence punctuation followed by a space or end-of-string.
# Requiring that space is what keeps "auth.py" and "app/main.py" intact - a
# bare '.' boundary cut every filename in half.
_CLAUSE_END = r'(?:\s*[.?!;](?:\s|$)|$)'
_STRIP = '`"\''
_STOPWORDS = ("the", "a", "an", "that", "this", "of", "file", "files", "directory")


def _clean_path(raw: str) -> str:
    """Reduce a path phrase to the token that actually names a path.

    "the auth module" is not a directory, but "module" is the last word of the
    phrase and is what the caller meant. Taking the first token instead yielded
    'the', and a wrong path is indistinguishable from a missing one downstream:
    Grep reports no match either way, so the agent just sees silence.
    """
    tokens = [t for t in raw.strip().split() if t.lower() not in _STOPWORDS]
    if not tokens:
        return ""
    return tokens[-1].strip(_STRIP)


def _grep_args(text: str) -> Optional[dict]:
    """Best-effort pattern/path out of a natural phrasing.

    This is a heuristic, not a parser: it hands a route to a tool the agent then
    runs itself, and a slightly-off path produces a "file not found" the agent
    can react to. Over-engineering it buys nothing and costs false positives.
    """
    # Quoted pattern first - unambiguous, and the only case worth trusting.
    #   grep "TODO" in src/  |  search for 'password' in auth.py
    m = re.search(rf'{_GREP_VERB}[^`"\']*?["\']([^"\']+)["\']'
                  rf'(?:\s+(?:in|under|within)\s+(.+?){_CLAUSE_END})?',
                  text, re.IGNORECASE)
    if m:
        pattern, path_raw = m.group(1), m.group(2)
    else:
        # Unquoted, split in two steps rather than one regex. A single pattern
        # with an optional trailing group makes (.+?) match the shortest thing
        # it can get away with - "grep TODO" parsed as pattern='T'. Ask for the
        # path form first; only if there is no path phrase, take the rest.
        m = re.search(rf'{_GREP_VERB}\s+(.+?)\s+(?:in|under|within)\s+(.+?){_CLAUSE_END}',
                      text, re.IGNORECASE)
        if m:
            pattern, path_raw = m.group(1), m.group(2)
        else:
            m = re.search(rf'{_GREP_VERB}\s+(.+?){_CLAUSE_END}', text, re.IGNORECASE)
            if not m:
                return None
            pattern, path_raw = m.group(1), None

    pattern = (pattern or "").strip()
    if not pattern:
        return None
    args = {"pattern": pattern}
    if path_raw:
        path = _clean_path(path_raw)
        if path:
            args["path"] = path
    return args


def _webfetch_args(text: str) -> Optional[dict]:
    # "fetch https://..." / "web fetch url"
    m = re.search(r'(https?://[^\s`"\'")\]]+)', text)
    if m and re.search(r'\b(?:fetch|web\s*fetch|open\s+url|visit)\b', text, re.IGNORECASE):
        return {"url": m.group(1)}
    return None


_INTENTS = {
    "Read": _read_args,
    "Write": _write_args,
    "Edit": _edit_args,
    "Bash": _bash_args,
    "Glob": _glob_args,
    "Grep": _grep_args,
    "WebFetch": _webfetch_args,
}


def _last_user_text(messages: List[dict]) -> str:
    """Extract the most recent user-role text from an Anthropic/OpenAI message list."""
    for m in reversed(messages):
        role = m.get("role", "")
        if role not in ("user", "tool"):
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        # a tool_result turn means the loop already executed; skip routing
                        return ""
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts)
        return ""
    return ""


def route_intent(messages: List[dict], tools: List[dict]) -> Optional[RoutedToolCall]:
    """Decide whether the latest user request maps to a registered tool call.

    `tools` is the client's tool list (Anthropic or OpenAI shape). Only tools whose
    name is in KNOWN_TOOLS and is actually offered by the client are routed.
    Returns a RoutedToolCall, or None to fall through to normal chat.
    """
    if not tools:
        return None
    client_tool_names = set()
    for t in tools:
        name = None
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            name = t["function"].get("name")
        else:
            name = t.get("name")
        if name:
            client_tool_names.add(name)

    text = _last_user_text(messages)
    if not text.strip():
        return None

    # Try known tools in priority order (Bash last so Read/Glob win on overlap).
    order = ["Read", "Glob", "Grep", "Write", "Edit", "WebFetch", "Bash"]
    for name in order:
        if name not in client_tool_names or name not in _INTENTS:
            continue
        args = _INTENTS[name](text)
        if args:
            return RoutedToolCall(name=name, input=args)
    return None


def has_tool_result(messages: List[dict]) -> bool:
    """True when the client already returned a tool_result (loop is mid-execution)."""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "tool":
            return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False
