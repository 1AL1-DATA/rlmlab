"""Opencode-style context compaction.

A dependency-free port of opencode's session compaction mechanics
(packages/opencode/src/session/compaction.ts) so any host (hermes RLM
gate, agent loops, the CLI) gets the same overflow behaviour:

1. **Overflow detection** — fire compaction when estimated tokens exceed
   ``context_length - buffer_tokens`` (the reserved output + safety buffer).
   An already-overflowed conversation therefore compacts on the very next
   check, no state required.
2. **Head/tail selection** — walk the history backward in full *turns* (a
   user message plus everything until the next user message). Keep as many
   recent turns as fit ``tail_token_budget``; if the oldest kept turn does
   not fully fit, split it so the newest portion is still kept verbatim.
   Everything older becomes the *head* that is summarized away.
3. **Serialization** — render messages as ``[User]/[Assistant]/[Assistant
   tool call]/[Tool result]`` text with tool/shell output truncated to
   ``tool_output_max_chars``.
4. **Running summary** — any prior summary message is folded into the head
   text so a later compaction *updates* the summary instead of stacking a
   fresh one. The LLM summarizer is injected as a callable (default: a
   deterministic slice, newest portion of the head kept verbatim).
5. **Pruning** — blank old completed tool output once accumulated output
   crosses ``prune_threshold_tokens``, reclaiming at least
   ``prune_reclaim_tokens``; protected tools are never blanked.

Message shape is the generic OpenAI-ish dict: ``{"role", "content"}`` where
``content`` is a string or a list of parts. Assistant messages may carry
``reasoning_content`` and ``tool_calls``; ``summary: True`` marks a prior
compaction summary message.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# -- token estimation --------------------------------------------------------

_PLAIN_TXT = "text"
_FILE_TXT = "file"
_MARKER_CLEARED = "[Old tool result content cleared]"
_SUMMARY_MARKERS = (
    "context summary",
    "compacted history",
    "condensed earlier conversation",
    "context gate:",
    "summary:",
)
_ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System update",
    "tool": "Tool result",
    "shell": "Shell",
    "synthetic": "Synthetic context",
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4), the same heuristic opencode uses."""
    return math.ceil((len(text) or 1) / 4)


def overflow_at(context_length: int, buffer_tokens: int) -> int:
    """Token count at which compaction must fire."""
    return max(0, context_length - buffer_tokens)


def should_compact(
    estimated_tokens: int, context_length: int, buffer_tokens: int
) -> bool:
    """True when the estimated request exceeds the usable window.

    Mirrors opencode's ``isOverflow()``: the reserved ``buffer_tokens`` keep
    output + overflow headroom. A conversation already over the line (e.g.
    reopened after growth) fires immediately on the next check. When the
    buffer covers the whole window there is no usable context, so compaction
    would be futile and never fires.
    """
    limit = overflow_at(context_length, buffer_tokens)
    return limit > 0 and estimated_tokens > limit


# -- configuration ------------------------------------------------------------

@dataclass
class CompactionConfig:
    """Budget knobs. Defaults target an 80K window with a 20K safety buffer
    and a guaranteed 20K verbatim tail (>=20K fallback after compaction)."""

    context_length: int = 80000
    buffer_tokens: int = 20000
    tail_token_budget: int = 20000
    tool_output_max_chars: int = 2000
    summary_target_tokens: int = 4096
    prune_threshold_tokens: int = 40000
    prune_reclaim_tokens: int = 20000
    prune_protected: tuple[str, ...] = ("skill",)
    # per-message serialization overhead in tokens
    per_message_tokens: int = 8


# -- serialization ------------------------------------------------------------

def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return __import__("json").dumps(value) if value is not None else ""
    except (TypeError, ValueError):
        return str(value)


def _text_of_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in (_PLAIN_TXT, _FILE_TXT) and part.get("text"):
                out.append(_to_str(part["text"]))
        return "\n".join(out)
    return _to_str(content)


def _reasoning_of(message: dict[str, Any]) -> str:
    return _to_str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    )


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else f"{text[:n]}\n[truncated]"


def _is_summary_message(message: dict[str, Any]) -> bool:
    if message.get("summary"):
        return True
    content = _text_of_content(message.get("content", "")).lower()
    return any(marker in content for marker in _SUMMARY_MARKERS)


def serialize_message(
    message: dict[str, Any], tool_output_max_chars: int = 2000
) -> str:
    """Render one message the way opencode renders it for compaction."""
    role = message.get("role", "")
    label = _ROLE_LABELS.get(role, role.capitalize())
    text = _text_of_content(message.get("content", ""))

    if role == "user":
        out = [f"[User]: {text}"] if text else []
        for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if isinstance(part, dict) and part.get("type") == _FILE_TXT:
                out.append(f"[Attached {part.get('mime', 'file')}: {part.get('filename', 'file')}]")
        return "\n".join(out)

    if role == "assistant":
        if _is_summary_message(message):
            return f"[Context summary]: {text}" if text else "[Context summary]"
        out: list[str] = []
        reasoning = _reasoning_of(message)
        if reasoning:
            out.append(f"[Assistant reasoning]: {_truncate(reasoning, 1000)}")
        if text:
            out.append(f"[Assistant]: {text}")
        for call in message.get("tool_calls", []) or []:
            if isinstance(call, dict):
                fn = call.get("function", {})
                name = _to_str(fn.get("name", call.get("name", "unknown")))
                args = _truncate(_to_str(fn.get("arguments", "")), 200)
                out.append(f"[Assistant tool call]: {name}({args})")
        return "\n".join(out)

    if role == "tool":
        name = _to_str(message.get("name", "tool"))
        if text == _MARKER_CLEARED:
            return f"[Tool result cleared]: {name}"
        return f"[Tool result]: {_truncate(text, tool_output_max_chars)}"

    if role == "shell":
        cmd = _to_str(message.get("command", ""))
        return f"[Shell]: {cmd}\n{_truncate(text, tool_output_max_chars)}"

    # system / synthetic / unknown
    return f"[{label}]: {text}" if text else f"[{label}]"


def serialize(
    messages: Sequence[dict[str, Any]], tool_output_max_chars: int = 2000
) -> str:
    return "\n".join(
        serialize_message(m, tool_output_max_chars) for m in messages
    )


def _message_tokens(
    message: dict[str, Any],
    tool_output_max_chars: int,
    per_message_tokens: int,
) -> int:
    return estimate_tokens(serialize_message(message, tool_output_max_chars)) + per_message_tokens


def _messages_tokens(
    messages: Sequence[dict[str, Any]],
    tool_output_max_chars: int,
    per_message_tokens: int,
) -> int:
    return sum(
        _message_tokens(m, tool_output_max_chars, per_message_tokens)
        for m in messages
    )


# -- head/tail selection -------------------------------------------------------

def _turns(messages: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    """Index ranges (start, end) of full turns; each turn starts at a user
    message and runs until the message before the next user message. Leading
    non-user messages (system prompt) form the first turn."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, m in enumerate(messages):
        if m.get("role") == "user":
            if start is not None:
                ranges.append((start, idx))
            start = idx
    if start is not None:
        ranges.append((start, len(messages)))
    elif messages:
        ranges.append((0, len(messages)))
    return ranges


def _tail_slice(
    messages: Sequence[dict[str, Any]],
    start: int,
    end: int,
    budget: int,
    tool_output_max_chars: int,
    per_message_tokens: int,
) -> tuple[int, int] | None:
    """Newest portion of ``[start, end)`` that fits ``budget`` tokens. Returns
    a (slice_start, end) range, or None when even one message cannot fit."""
    used = 0
    idx = end
    while idx > start:
        cost = _message_tokens(messages[idx - 1], tool_output_max_chars, per_message_tokens)
        if used + cost > budget:
            break
        used += cost
        idx -= 1
    if idx == end:
        return None
    # Never open the tail on a bare tool result whose tool call lives in the
    # head — snap forward to the message that pairs with it.
    while idx < end and messages[idx].get("role") == "tool":
        idx += 1
    if idx == end:
        return None
    return idx, end


def select(
    messages: Sequence[dict[str, Any]],
    tail_token_budget: int,
    tool_output_max_chars: int = 2000,
    per_message_tokens: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``messages`` into ``(head, tail)``.

    The tail holds the most recent turns kept verbatim within
    ``tail_token_budget`` (the oldest kept turn may be partial). The head is
    everything older and is what a summarizer condenses away.
    """
    budget = max(int(tail_token_budget), 1)
    turns = _turns(messages)
    tail_start = 0
    used = 0
    for (s, e) in reversed(turns):
        cost = _messages_tokens(
            messages[s:e], tool_output_max_chars, per_message_tokens
        )
        if tail_start != 0 and used + cost > budget:
            split = _tail_slice(
                messages, s, e, budget - used, tool_output_max_chars, per_message_tokens
            )
            if split is not None:
                tail_start = split[0]
            break
        tail_start = s
        used += cost
    return list(messages[:tail_start]), list(messages[tail_start:])


# -- pruning ------------------------------------------------------------------

def _tool_name_of(message: dict[str, Any]) -> str:
    name = message.get("name", "")
    if name:
        return str(name)
    return "tool"


def prune(
    messages: list[dict[str, Any]],
    config: CompactionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Blank old completed tool output (not a re-summarize).

    Walks oldest-first, replacing tool-result content with a cleared marker
    until accumulated output crosses ``prune_threshold_tokens`` and at least
    ``prune_reclaim_tokens`` are reclaimed. Tools named in
    ``prune_protected`` are skipped. Messages are copied, never mutated.
    """
    protected = set(config.prune_protected)
    results: list[dict[str, Any]] = []
    cleared = 0
    out: list[dict[str, Any]] = []
    for m in messages:
        if (
            m.get("role") == "tool"
            and _tool_name_of(m) not in protected
            and m.get("content")
            and _text_of_content(m.get("content")) != _MARKER_CLEARED
        ):
            raw = _text_of_content(m.get("content"))
            results.append(raw)
            if (
                estimate_tokens(" ".join(results)) >= config.prune_threshold_tokens
                and cleared < config.prune_reclaim_tokens
            ):
                copy = dict(m)
                copy["content"] = _MARKER_CLEARED
                copy["pruned"] = True
                cleared += estimate_tokens(raw) - estimate_tokens(_MARKER_CLEARED)
                out.append(copy)
                continue
        out.append(m)
    return out, {
        "pruned": cleared > 0,
        "cleared_results": cleared,
        "blanked_tool_results": sum(
            1 for m in out if m.get("role") == "tool" and m.get("pruned")
        ),
    }


# -- summary ----------------------------------------------------------------

def deterministic_summary(
    head: list[dict[str, Any]],
    head_text: str,
    config: CompactionConfig,
) -> str:
    """Fallback when no LLM summarizer is available: keep the newest slice of
    the head verbatim behind a condensed marker, mirroring the double-compact
    deterministic slice."""
    budget = max(config.tail_token_budget // 2, 256)
    kept: list[str] = []
    used = 0
    for m in reversed(head):
        rendered = serialize_message(m, config.tool_output_max_chars)
        est = estimate_tokens(rendered) + config.per_message_tokens
        if kept and used + est > budget:
            break
        kept.insert(0, rendered)
        used += est
    body = "\n".join(kept) if kept else head_text[: int(budget * 4)]
    return f"[Condensed earlier conversation]\n{_truncate(body, budget * 4)}"


def fold_prior_summary(head: list[dict[str, Any]]) -> str:
    """Return the newest prior summary text already inside the head, so a new
    summary *updates* the running one instead of restacking it."""
    for m in reversed(head):
        if _is_summary_message(m):
            return _text_of_content(m.get("content", "")).strip()
    return ""


# -- orchestration ------------------------------------------------------------

Summarizer = Callable[[str, str], str]  # (head_text, prior_summary) -> summary


def compact(
    messages: list[dict[str, Any]],
    config: CompactionConfig | None = None,
    summarize: Summarizer | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compaction entry point: overflow-checked head summarization.

    When the request is under the overflow line nothing changes. Otherwise the
    head (everything older than the verbatim tail) is replaced by a single
    summary message; the tail is preserved verbatim. ``summarize`` is an
    optional callable ``(head_text, prior_summary) -> summary_text``; when
    omitted the deterministic slice is used. Messages are copied, never
    mutated.
    """
    cfg = config or CompactionConfig()
    total = _messages_tokens(messages, cfg.tool_output_max_chars, cfg.per_message_tokens)
    if not should_compact(total, cfg.context_length, cfg.buffer_tokens):
        return list(messages), {
            "compacted": False,
            "reason": "under_budget",
            "tokens_before": total,
        }

    head, tail = select(
        messages, cfg.tail_token_budget, cfg.tool_output_max_chars, cfg.per_message_tokens
    )
    if not head:
        return list(messages), {
            "compacted": False,
            "reason": "tail_covers_history",
            "tokens_before": total,
        }

    head_text = serialize(head, cfg.tool_output_max_chars)
    prior = fold_prior_summary(head)
    if summarize is not None:
        try:
            summary = summarize(head_text, prior)
        except Exception:  # noqa: BLE001 - summarizer failure must degrade, not throw
            summary = deterministic_summary(head, head_text, cfg)
    else:
        summary = deterministic_summary(head, head_text, cfg)

    summary_message: dict[str, Any] = {
        "role": "assistant",
        "summary": True,
        "content": summary,
    }
    compacted = [summary_message, *tail]
    after = _messages_tokens(compacted, cfg.tool_output_max_chars, cfg.per_message_tokens)
    return compacted, {
        "compacted": True,
        "method": "llm" if summarize is not None else "deterministic",
        "trigger_tokens": total,
        "overflow_at": overflow_at(cfg.context_length, cfg.buffer_tokens),
        "tokens_before": total,
        "tokens_after": after,
        "head_messages": len(head),
        "tail_messages": len(tail),
        "summary_tokens": estimate_tokens(summary),
    }