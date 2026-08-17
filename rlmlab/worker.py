"""RLM worker loop: drains admitted subagents.

Phase 1 (deterministic executor): each admitted subagent gets a dedicated
kernel session named after it, boots with a compact distilled note block
injected from the continual harness (hybrid memory), then executes its
prompt as code. Results are delivered to the child's mailbox and its status
flips to done/failed.

Crash-safe: agents are claimed (admitted -> running) before processing and
a non-blocking flock guards the whole drain, so overlapping workers never
double-process (claim-before-effects).
"""

import fcntl
import json
import os
import re
import time

from . import agent_loop, apis, goals, harness, kernel, subagents

LOCK_FILE = os.path.join(subagents.HOME, "worker.lock")

MAX_RESULT_LENGTH = 64_000
DEFAULT_TIMEOUT = 120

WRANGLER_SYSTEM = (
    "You are a WRANGLER: an operator agent with live access to the rlmlab task "
    "queue. You decide which admitted tasks to work on and how, then you execute "
    "them yourself. Tasks are objects with a child id, name, api executor "
    "(deterministic = plain Python code, llama/ollama = LLM agent) and depth.\n"
    "The kernel has these helpers, call them with exec actions:\n"
    "  queue()                  -> admitted tasks: child, name, api, created, depth\n"
    "  inspect(child)           -> full record + prompt + latest result of any task\n"
    "  child_status(child)      -> status/result of any task\n"
    "  claim_and_run(child, timeout=90) -> claim ONE admitted task and run it\n"
    "      synchronously right now; returns {'ok', 'child', 'text'|'error'}. This\n"
    "      is how you 'pick a task from the queue and do it'. Deterministic tasks\n"
    "      are fast; keep each claim under ~90s.\n"
    "  send(child, text)        -> push a message into a task's mailbox\n"
    "  run_child / child_result -> spawn and track new work as usual\n"
    "Limits: 12 claim_and_run per wrangler run, 90s each, total wall-clock budget "
    "~15 minutes.\n"
    "The strategy is yours: pick tasks in any order, skip some, message hints to "
    "others, spawn helpers. When done, reply final with a concise summary of what "
    "you processed, skipped, and why."
)

WRANGLER_PREAMBLE = """
_WRANGLER_USED = {"n": 0}

def queue():
    return [{"child": r["rlm_child_id"], "name": r.get("name"), "api": r.get("api"),
             "created": r.get("created"), "depth": r.get("depth")}
            for r in _sa.list_subagents() if r.get("status") == "admitted"]

def inspect(child_id):
    rec = _sa._find(child_id)
    if rec is None:
        return {"ok": False, "error": f"no task {child_id}"}
    out = dict(rec)
    out["prompt"] = _sa.prompt(child_id)
    out["result"] = _sa.child_result(child_id)
    return out

def child_status(child_id):
    return _sa.child_result(child_id)

def send(child_id, text):
    return _sa.send(child_id, text)

def claim_and_run(child_id, timeout=90):
    if _WRANGLER_USED["n"] >= 12:
        return {"ok": False, "error": "wrangler cap reached (12 claim_and_run per run)"}
    claimed = _sa.claim(child_id)
    if claimed is None:
        return {"ok": False, "error": f"cannot claim {child_id}: not admitted (running/done/failed)"}
    _WRANGLER_USED["n"] += 1
    from rlmlab import worker as _w
    return _w._process_one(claimed, min(int(timeout), 90), "deterministic", {})
"""


_MEMORY_BUDGET_CHARS = 8_000
_MEMORY_K = 20


def _retrieve_memories(memories, query, k=_MEMORY_K, max_chars=_MEMORY_BUDGET_CHARS):
    """Top-k memories by keyword overlap with the task prompt, capped at a
    token budget. Without this the harness dumps up to MEMORY_CAP entries
    into every child, which is both a token firehose and undifferentiated
    context. Scoring is bag-of-words overlap — deliberately simple."""
    if not memories:
        return []
    tokens = {w for w in re.findall(r"[a-zA-Z0-9_]{3,}", query.lower())}
    if not tokens:
        return memories[:k]
    scored = []
    for m in memories:
        body = m.lower() if isinstance(m, str) else ""
        hit = sum(1 for t in tokens if t in body)
        if hit:
            scored.append((hit, len(body), m))
    picked, used = [], 0
    for _hit, _len, m in sorted(scored, reverse=True):
        if used + len(m) > max_chars:
            break
        picked.append(m)
        used += len(m)
    return picked or memories[:k]


def _distill_notes(query=None):
    state = harness.get_state()
    notes = {}
    for section in ("prompt_notes", "skill_descriptions", "subagent_specs"):
        entries = state.get(section, [])
        notes[section] = [e.get("text") if isinstance(e, dict) else e for e in entries]
    notes["memories"] = _retrieve_memories(
        [e.get("text") if isinstance(e, dict) else e for e in state.get("memories", [])],
        query or "",
    )
    notes["goals"] = [g["text"] for g in goals.list_goals(status="open")]
    return notes


def _inject_preamble(notes, child_id=None, depth=0, executor="deterministic", timeout=120, max_depth=3, wrangler=False):
    encoded = json.dumps(notes, ensure_ascii=False)
    parent = "None" if child_id is None else repr(child_id)
    preamble = (
        f"import json\nNOTES = {encoded}\n"
        f"from rlmlab import subagents as _sa\n"
        f"_CHILD_ID = {parent}\n"
        f"_CHILD_DEPTH = {int(depth)}\n"
        f"_CHILD_MAX_DEPTH = {int(max_depth)}\n"
        f"_CHILD_EXECUTOR = {executor!r}\n"
        f"_CHILD_TIMEOUT = {int(timeout)}\n"
        "def run_child(name, prompt):\n"
        "    if _CHILD_DEPTH + 1 > _CHILD_MAX_DEPTH:\n"
        '        return {"ok": False, "error": "recursion depth limit reached"}\n'
        "    return _sa.run(name, prompt, parent=_CHILD_ID, depth=_CHILD_DEPTH + 1)['rlm_child_id']\n"
        "def child_result(child_id, wait=120):\n"
        "    import time as _t\n"
        "    r = _sa.child_result(child_id)\n"
        "    if r.get('status') in ('admitted', 'running') and wait > 0:\n"
        "        from rlmlab import worker as _w\n"
        "        _w._wait_child(child_id, _CHILD_TIMEOUT, _CHILD_EXECUTOR, {}, 1)\n"
        "        r = _sa.child_result(child_id)\n"
        "    return r\n"
    )
    if wrangler:
        preamble += WRANGLER_PREAMBLE
    return preamble


def _distill_feedback(record, ok, text):
    """Close the RL loop: write the outcome back into the harness so the
    next run starts cumulatively smarter. Successes become reusable
    memories; failures become negative examples to avoid."""
    name = record.get("name", record.get("rlm_child_id"))
    if ok:
        snippet = text[:160].replace("\n", " ")
        note = f"subagent '{name}' completed: {snippet}"
    else:
        snippet = text[:160].replace("\n", " ")
        note = f"subagent '{name}' failed: {snippet}"
    try:
        harness.refine("memories", note)
    except (OSError, ValueError):
        pass  # best-effort distillation; a failed memory write must not fail the subagent


def _wait_child(child_id, timeout, executor, llm_opts, _depth, max_depth=None):
    """Synchronously drain a pending child so a parent can block on it."""
    if max_depth is None:
        max_depth = llm_opts.get("max_depth", agent_loop.DEFAULT_MAX_DEPTH)
    if _depth > max_depth:
        return subagents.child_result(child_id)
    deadline = time.time() + min(llm_opts.get("max_seconds", 300), 600)
    while time.time() < deadline:
        status = subagents.child_result(child_id).get("status")
        if status in ("done", "failed"):
            break
        if status == "admitted":
            claimed = subagents.claim(child_id)
            if claimed is not None:
                _process_one(claimed, timeout, executor, llm_opts, _depth + 1)
        time.sleep(1)
    return subagents.child_result(child_id)


def _process_one(record, timeout, executor, llm_opts, _depth=0):
    child_id = record["rlm_child_id"]
    prompt_text = subagents.prompt(child_id)
    if prompt_text is None:
        subagents.complete(child_id, "", error="missing prompt.txt")
        return {"ok": False, "error": "missing prompt.txt"}

    resolved = apis.resolve(record.get("api") or "deterministic") or {"executor": "deterministic"}
    executor = resolved.get("executor", executor)
    llm_opts = {**(llm_opts or {})}
    for key in ("base_url", "model"):
        if resolved.get(key):
            llm_opts.setdefault(key, resolved[key])
    wrangler = bool(record.get("wrangler"))

    started = kernel.start(child_id)
    if started.get("ok") is False:
        error = started.get("error", "kernel start failed")
        subagents.complete(child_id, "", error=error)
        return {"ok": False, "child": child_id, "error": error}

    try:
        if executor == "llm":
            max_depth = llm_opts.get("max_depth", agent_loop.DEFAULT_MAX_DEPTH)
            notes_ok = kernel.exec_code(
                child_id,
                _inject_preamble(
                    _distill_notes(prompt_text), child_id, record.get("depth", 0), "llm", timeout, max_depth, wrangler=wrangler
                ),
                timeout=30,
            )
            if notes_ok.get("ok") is False:
                subagents.complete(child_id, "", error=notes_ok.get("error", "notes injection failed"))
                _distill_feedback(record, False, "notes injection failed")
                return {"ok": False, "child": child_id, "error": "notes injection failed"}
            result = agent_loop.run_llm(
                prompt_text,
                session=child_id,
                base_url=llm_opts.get("base_url", agent_loop.DEFAULT_BASE_URL),
                model=llm_opts.get("model", agent_loop.DEFAULT_MODEL),
                max_turns=llm_opts.get("max_turns", agent_loop.DEFAULT_MAX_TURNS),
                max_seconds=llm_opts.get("max_seconds", agent_loop.DEFAULT_MAX_SECONDS)
                * (3 if wrangler else 1),
                exec_timeout=max(timeout, 300) if wrangler else timeout,
                child_id=child_id,
                depth=record.get("depth", 0),
                max_depth=max_depth,
                wait_child=lambda cid: _wait_child(cid, timeout, executor, llm_opts, _depth + 1),
                extra_system=WRANGLER_SYSTEM if wrangler else None,
            )
            if result.get("ok") is False:
                subagents.complete(child_id, "", error=result.get("error", "llm execution failed"))
                _distill_feedback(record, False, result.get("error", "llm execution failed"))
                return {"ok": False, "child": child_id, "error": result["error"][:1000]}
            subagents.complete(
                child_id,
                result["answer"],
                meta={"turns": result.get("turns"), "tokens": result.get("tokens")},
            )
            _distill_feedback(record, True, result["answer"])
            return {"ok": True, "child": child_id, "text": result["answer"], "turns": result["turns"]}

        code = _inject_preamble(
            _distill_notes(prompt_text), child_id, record.get("depth", 0), "deterministic", timeout, agent_loop.DEFAULT_MAX_DEPTH,
            wrangler=wrangler,
        ) + "\n" + prompt_text
        result = kernel.exec_code(child_id, code, timeout=timeout)
        if result.get("ok") is False:
            subagents.complete(child_id, "", error=result.get("error", "execution failed"))
            return {"ok": False, "child": child_id, "error": result["error"][:1000]}
        text = result.get("text", "")
        if len(text) > MAX_RESULT_LENGTH:
            text = text[:MAX_RESULT_LENGTH] + "\n...[truncated]"
        subagents.complete(child_id, text)
        return {"ok": True, "child": child_id, "text": text}
    except Exception as e:  # noqa: BLE001 - any failure must mark the child failed, never strand it
        error = f"{type(e).__name__}: {e}"
        subagents.complete(child_id, "", error=error)
        return {"ok": False, "child": child_id, "error": error[:1000]}
    finally:
        kernel.stop(child_id)


def supervise(interval=5, timeout=DEFAULT_TIMEOUT, executor="llm", llm_opts=None, api=None):
    """Run the drain loop forever until SIGTERM/SIGINT."""
    stop = {"flag": False}

    def _handler(_signum, _frame):
        stop["flag"] = True

    import signal

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    while not stop["flag"]:
        result = work_once(timeout=timeout, executor=executor, llm_opts=llm_opts, api=api)
        if result.get("ok") and result["processed"]:
            for r in result["processed"]:
                line = f"  ok   {r.get('child')}: {r.get('text','')[:60]}" if r.get("ok") else f"  FAIL {r.get('child')}: {r.get('error','')[:60]}"
                print(line, flush=True)
        time.sleep(interval)
    return {"ok": True, "stopped": "supervisor"}


def reap(max_age_seconds=3600):
    """Stop kernels idle longer than max_age that aren't attached to an
    admitted/running subagent; drop dead-pid index files."""
    active = {
        r["rlm_child_id"]
        for r in subagents.list_subagents()
        if r.get("status") in ("admitted", "running")
    }
    now = int(time.time() * 1000)
    reaped = []
    stale = []
    for s in kernel.list_sessions():
        if not s.get("alive"):
            stale.append(s["name"])
            kernel.stop(s["name"])
            continue
        if s["name"] in active:
            continue
        age = now - s.get("last_active", 0)
        if age >= max_age_seconds * 1000:
            reaped.append(s["name"])
            kernel.stop(s["name"])
    return {"ok": True, "reaped": reaped, "stale": stale}


def _requeue_stale_running(max_age_seconds=600):
    """Reset children stuck in 'running' whose kernel is gone and whose
    claim is older than max_age back to 'admitted' so a future drain picks
    them up again. A worker killed mid-run (kill -9, crash) leaves a child
    claimed-but-never-completed; without this it is stranded forever."""
    now = int(time.time() * 1000)
    requeued = []
    for r in subagents.list_subagents():
        if r.get("status") != "running":
            continue
        started_ts = r.get("started_ts") or 0
        if now - started_ts < max_age_seconds * 1000:
            continue
        idx = kernel._read_index(r["rlm_child_id"])
        if idx is not None and kernel._pid_alive(idx):
            continue  # a live kernel means a worker is genuinely still processing it
        subagents.requeue(r["rlm_child_id"])
        requeued.append(r["rlm_child_id"])
    return requeued


def work_once(limit=None, timeout=DEFAULT_TIMEOUT, executor="deterministic", llm_opts=None, api=None):
    lock = open(LOCK_FILE, "w")  # noqa: SIM115 - lock handle must stay open across the whole function
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return {"ok": False, "error": "another worker is already running", "processed": []}

    processed = []
    try:
        requeued = _requeue_stale_running()
        admitted = [r for r in subagents.list_subagents() if r.get("status") == "admitted"]
        if api:
            admitted = [r for r in admitted if r.get("api") == api]
        for record in admitted[:limit] if limit else admitted:
            claimed = subagents.claim(record["rlm_child_id"])
            if claimed is None:
                continue
            result = _process_one(claimed, timeout, executor, llm_opts or {})
            processed.append(result)
        return {"ok": True, "processed": processed, "requeued": requeued}
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
