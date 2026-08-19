"""Tests for opencode-style compaction (rlmlab.compaction)."""

from rlmlab import compaction as c


def _conv(turns=20, text_len=6000):
    """A synthetic conversation: `turns` user/assistant pairs + a system msg."""
    msgs = [{"role": "system", "content": "you are a helper"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"question {i}: " + "x" * text_len})
        msgs.append({"role": "assistant", "content": f"answer {i}: " + "y" * text_len})
    return msgs


def test_should_compact_overflow_boundary():
    # 80K window, 20K buffer -> fire at 60K
    assert c.overflow_at(80000, 20000) == 60000
    assert c.should_compact(59999, 80000, 20000) is False
    assert c.should_compact(60000, 80000, 20000) is False
    assert c.should_compact(60001, 80000, 20000) is True
    # degenerate: buffer covers the whole window -> never fire
    assert c.should_compact(1 << 20, 80000, 80000) is False


def test_already_overflowed_conversation_compacts_immediately():
    """Opening an already-overflowed conversation must compact on the first
    check (the hermes use case: reopen a session that outgrew its window)."""
    msgs = _conv(turns=20)
    cfg = c.CompactionConfig(context_length=80000, buffer_tokens=20000, tail_token_budget=20000)
    total = c._messages_tokens(msgs, cfg.tool_output_max_chars, cfg.per_message_tokens)
    assert total > c.overflow_at(80000, 20000)

    out, meta = c.compact(msgs, cfg, summarize=lambda head, prior: "SUMMARY")
    assert meta["compacted"] is True
    assert out[0]["role"] == "assistant" and out[0].get("summary") is True
    assert out[0]["content"] == "SUMMARY"


def test_select_keeps_recent_tail_verbatim():
    msgs = _conv(turns=20)
    cfg = c.CompactionConfig(tail_token_budget=20000)
    head, tail = c.select(msgs, cfg.tail_token_budget, cfg.tool_output_max_chars, cfg.per_message_tokens)
    assert head and tail
    tail_tokens = c._messages_tokens(tail, cfg.tool_output_max_chars, cfg.per_message_tokens)
    assert tail_tokens <= cfg.tail_token_budget * 1.5  # allow the oldest kept turn to straddle
    # tail keeps the newest messages verbatim (not summarized)
    assert tail[-1] == msgs[-1]


def test_select_splits_oldest_turn_and_never_opens_on_tool_result():
    msgs = [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
        {"role": "tool", "name": "read_file", "content": "t0"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    # tiny budget: only the newest turn (user+assistant) can fit
    _, tail = c.select(msgs, 4, 2000, 8)
    assert tail and tail[0]["role"] == "user"
    assert not any(m["role"] == "tool" for m in tail)


def test_fold_prior_summary_running_update():
    head = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "summary": True, "content": "PRIOR SUMMARY"},
    ]
    assert c.fold_prior_summary(head) == "PRIOR SUMMARY"


def test_compact_folds_prior_summary_into_head_text():
    msgs = _conv(turns=20)
    # prior summary sits mid-conversation (older than the tail budget) so it
    # falls into the head and must be folded in
    msgs.insert(20, {"role": "assistant", "summary": True, "content": "EARLIER SUMMARY"})
    cfg = c.CompactionConfig(context_length=80000, buffer_tokens=20000, tail_token_budget=20000)
    seen = {}

    def summarize(head_text, prior):
        seen["prior"] = prior
        return "NEW SUMMARY"

    _, meta = c.compact(msgs, cfg, summarize=summarize)
    assert meta["compacted"] is True
    assert seen["prior"] == "EARLIER SUMMARY"


def test_prune_blanks_old_output_and_protects_skill():
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append(
            {
                "role": "assistant",
                "content": f"a{i}",
                "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}],
            }
        )
        msgs.append({"role": "tool", "name": "grep", "content": "Z" * 40000})
    msgs.append({"role": "tool", "name": "skill", "content": "S" * 40000})

    cfg = c.CompactionConfig(prune_threshold_tokens=40000, prune_reclaim_tokens=20000)
    out, meta = c.prune(msgs, cfg)
    assert meta["pruned"] is True
    assert meta["blanked_tool_results"] >= 1
    cleared = [m for m in out if m.get("role") == "tool" and m.get("pruned")]
    assert all(m["name"] == "grep" for m in cleared)
    skill = next(m for m in out if m.get("role") == "tool" and m.get("name") == "skill")
    assert skill.get("pruned") is not True


def test_compact_under_budget_is_a_noop():
    msgs = [{"role": "user", "content": "short"}]
    cfg = c.CompactionConfig(context_length=80000, buffer_tokens=20000)
    out, meta = c.compact(msgs, cfg, summarize=lambda h, p: "x")
    assert meta["compacted"] is False
    assert meta["reason"] == "under_budget"
    assert out == msgs


def test_compact_deterministic_fallback_no_llm():
    msgs = _conv(turns=20)
    cfg = c.CompactionConfig(context_length=80000, buffer_tokens=20000, tail_token_budget=20000)
    out, meta = c.compact(msgs, cfg)  # no summarizer
    assert meta["compacted"] is True
    assert meta["method"] == "deterministic"
    assert "[Condensed earlier conversation]" in out[0]["content"]