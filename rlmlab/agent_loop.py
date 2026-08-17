"""LLM executor (Phase 2): a real model loop driving a child kernel.

The model's one tool is the child's IPython kernel, exposed over a simple
JSON action protocol that local models handle far more reliably than native
function calling:

    {"action": "exec", "code": "..."}   -> run in the kernel, feed output back
    {"action": "final", "answer": "..."} -> task complete, deliver the answer

Bounded by quality gates: max turns, wall-clock budget, per-turn kernel
output cap, and a max completion size.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from . import kernel, subagents

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "kat-coder:q6_k"
DEFAULT_MAX_TURNS = 12
DEFAULT_MAX_SECONDS = 300
DEFAULT_MAX_DEPTH = 3
MAX_TOKENS_PER_TURN = 1024
MAX_OUTPUT_PER_TURN = 6_000

SYSTEM_PROMPT = (
    "You are a subagent inside a persistent IPython kernel (Python 3.12). "
    "State you define (variables, dataframes, imports) persists between your turns. "
    "A dict named NOTES is preloaded in the kernel with durable notes: "
    "keys are prompt_notes, memories, skill_descriptions, subagent_specs. "
    "Do your work by running code in the kernel.\n"
    "You may also delegate subtasks to your own subagents. Children are "
    "processed asynchronously by a worker; poll child_result until status is "
    "'done' (status 'failed' means it errored; check its error flag).\n"
    "Each turn reply with EXACTLY ONE JSON object, no markdown, no prose, in one of these shapes:\n"
    '  {"action":"exec","code":"<python code>"}   -- run code in the kernel\n'
    '  {"action":"run_child","name":"<label>","prompt":"<task for the child>"} -- delegate a subtask (async)\n'
    '  {"action":"child_result","child":"<rlm_child_id>"} -- check a delegated child (status + result)\n'
    '  {"action":"list_children"}                 -- list the children you have delegated\n'
    '  {"action":"final","answer":"<result>"}      -- task complete; answer is the final concise result\n'
)


def _extract_json(content):
    """Extract JSON from model output. Handles markdown code blocks and
    falls back to bracket-search for mangled output."""
    if content is None:
        return None
    text = content.strip()
    # Strip markdown code block
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:]).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find the largest valid JSON object
    for start_idx, c1 in enumerate(text):
        if c1 == "{" or c1 == "[":
            for end_idx in range(len(text) - 1, start_idx, -1):
                if text[end_idx] in "}]":
                    try:
                        return json.loads(text[start_idx:end_idx + 1])
                    except json.JSONDecodeError:
                        pass
    return None


def _complete(base_url, model, messages, timeout=120):
    if urllib.parse.urlparse(base_url).scheme not in ("http", "https"):
        raise ValueError(f"unsupported base_url scheme: {base_url!r}")
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_TOKENS_PER_TURN,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - scheme validated above (http/https only)
        data = json.loads(resp.read())
    message = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return (
        message.get("content") or "",
        message.get("reasoning_content") or "",
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


def run_llm(
    prompt,
    session,
    base_url=DEFAULT_BASE_URL,
    model=DEFAULT_MODEL,
    max_turns=DEFAULT_MAX_TURNS,
    max_seconds=DEFAULT_MAX_SECONDS,
    exec_timeout=120,
    child_id=None,
    depth=0,
    max_depth=DEFAULT_MAX_DEPTH,
    wait_child=None,
    extra_system=None,
):
    system = SYSTEM_PROMPT + ("\n\n" + extra_system) if extra_system else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    start = time.time()
    total_tokens = 0

    for turn in range(max_turns):
        if time.time() - start > max_seconds:
            return {"ok": False, "error": f"wall-clock budget exceeded ({max_seconds}s)"}

        try:
            remaining = max_seconds - (time.time() - start)
            call_timeout = min(300, max(30, remaining))
            content, reasoning, p_in, p_out = _complete(base_url, model, messages, timeout=call_timeout)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            return {"ok": False, "error": f"model call failed: {e}"}
        total_tokens += p_in + p_out

        action = _extract_json(content) or _extract_json(reasoning)
        if action is None:
            messages.append({"role": "assistant", "content": content or reasoning or ""})
            messages.append(
                {"role": "user", "content": "Your reply was not valid JSON. Reply with exactly one JSON object."}
            )
            continue

        if action.get("action") == "final":
            answer = str(action.get("answer", "")).strip()
            if not answer:
                messages.append({"role": "user", "content": "final answer was empty; give a concise answer."})
                continue
            return {"ok": True, "answer": answer, "turns": turn + 1, "tokens": total_tokens}

        if action.get("action") == "exec":
            code = action.get("code")
            if not code:
                messages.append({"role": "user", "content": "exec action missing 'code'."})
                continue
            result = kernel.exec_code(session, code, timeout=exec_timeout)
            output = result.get("text", "") if result.get("ok") else result.get("error", "unknown error")
            if len(output) > MAX_OUTPUT_PER_TURN:
                output = output[:MAX_OUTPUT_PER_TURN] + "\n...[truncated]"
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Kernel output:\n{output}"})
            continue

        if action.get("action") == "run_child":
            if child_id is None:
                messages.append({"role": "user", "content": "run_child unavailable: no parent context."})
                continue
            if depth + 1 > max_depth:
                messages.append(
                    {"role": "user", "content": f"run_child rejected: recursion depth limit {max_depth} reached."}
                )
                continue
            name = str(action.get("name", "subtask"))[:64]
            task = str(action.get("prompt", ""))
            if not task:
                messages.append({"role": "user", "content": "run_child missing 'prompt'."})
                continue
            created = subagents.run(name, task, parent=child_id, depth=depth + 1)
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "delegated": True,
                            "child": created["rlm_child_id"],
                            "status": created["status"],
                            "depth": created["depth"],
                            "note": "children run asynchronously; poll child_result until done.",
                        }
                    ),
                }
            )
            continue

        if action.get("action") == "child_result":
            target = str(action.get("child", ""))
            if not target:
                messages.append({"role": "user", "content": "child_result missing 'child'."})
                continue
            res = subagents.child_result(target)
            if res.get("ok") and res.get("status") in ("admitted", "running") and wait_child is not None:
                wait_child(target)
                res = subagents.child_result(target)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": json.dumps(res, ensure_ascii=False)})
            continue

        if action.get("action") == "list_children":
            mine = [r for r in subagents.list_subagents() if r.get("parent") == child_id]
            summary = [{"child": r["rlm_child_id"], "name": r.get("name"), "status": r.get("status")} for r in mine]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": json.dumps(summary, ensure_ascii=False)})
            continue

        messages.append({"role": "user", "content": f"Unknown action: {action}. Use exec, final, run_child, child_result or list_children."})

    return {"ok": False, "error": f"max turns exceeded ({max_turns})"}
