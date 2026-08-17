"""RLM recursive subagents: admission handles + async mailboxes.

Matches prime-agent's invariant: rlm(...) returns an admission handle
immediately (never the child's answer); results arrive later as messages.
"""

import fcntl
import json
import os
import time
import uuid

HOME = os.environ.get("RLMLAB_HOME", os.path.expanduser("~/.rlmlab"))
SUBAGENTS_DIR = os.path.join(HOME, "subagents")

# ponytail: in-memory index for list_subagents; rebuild every 5s
_list_cache = {"data": None, "ts": 0}
_LIST_CACHE_TTL_MS = 5_000


def _bust_list_cache():
    _list_cache["data"] = None
    _list_cache["ts"] = 0


def _ensure():
    os.makedirs(SUBAGENTS_DIR, exist_ok=True)


def run(name, prompt, parent=None, depth=0, api=None, wrangler=False):
    _bust_list_cache()
    _ensure()
    child_id = "rlm_" + uuid.uuid4().hex[:16]
    session_dir = os.path.join(SUBAGENTS_DIR, child_id)
    os.makedirs(session_dir, exist_ok=True)
    record = {
        "rlm_child_id": child_id,
        "name": name,
        "session_dir": session_dir,
        "status": "admitted",
        "depth": int(depth),
        "parent": parent,
        "api": api or os.environ.get("RLMLAB_DEFAULT_API") or "deterministic",
        "created": int(time.time() * 1000),
        "wrangler": bool(wrangler),
    }
    with open(os.path.join(session_dir, "admission.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(session_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(session_dir, "mailbox.jsonl"), "a"):
        pass
    return {"ok": True, **record}


def send(child_id, text, role="agent"):
    record = _find(child_id)
    if record is None:
        return {"ok": False, "error": f"no subagent {child_id}"}
    message = {
        "from": role,
        "text": text,
        "ts": int(time.time() * 1000),
    }
    with open(os.path.join(record["session_dir"], "mailbox.jsonl"), "a") as f:
        f.write(json.dumps(message) + "\n")
    return {"ok": True, "message": message}


def claim(child_id):
    """Atomically transition admitted -> running.

    Returns the claimed record, or None if the agent is not 'admitted'
    (already running, done, or failed) -- so a crashed or re-run worker
    can never double-process an agent (claim-before-effects).

    The transition is a read-modify-write on admission.json, so it takes a
    per-child flock to be atomic even without the worker's global lock
    (two workers, or a worker + manual CLI racing).
    """
    lock_path = os.path.join(SUBAGENTS_DIR, child_id, "claim.lock")
    try:
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record = _find(child_id)
            if record is None or record.get("status") != "admitted":
                return None
            record["status"] = "running"
            record["started_ts"] = int(time.time() * 1000)
            _write(record)
            fcntl.flock(lock, fcntl.LOCK_UN)
            _bust_list_cache()
            return record
    except OSError:
        return None  # another claim is in flight; treat as already claimed


def set_api(child_id, api):
    """Reassign which API/executor should process an admitted subagent."""
    record = _find(child_id)
    if record is None:
        return {"ok": False, "error": f"no subagent {child_id}"}
    if record.get("status") != "admitted":
        return {"ok": False, "error": f"cannot reassign api: status is {record.get('status')!r} (only 'admitted' tasks can be reassigned)"}
    record["api"] = api or "deterministic"
    record["api_updated"] = int(time.time() * 1000)
    _write(record)
    return {"ok": True, "child": child_id, "api": record["api"]}


def requeue(child_id):
    """Reset a stuck 'running' child back to 'admitted' so a future drain
    re-claims it. Used by the stale-running recovery in worker.work_once."""
    record = _find(child_id)
    if record is None:
        return {"ok": False, "error": f"no subagent {child_id}"}
    if record.get("status") != "running":
        return {"ok": False, "error": f"cannot requeue: status is {record.get('status')!r}"}
    record["status"] = "admitted"
    record["started_ts"] = None
    record.pop("error", None)
    _write(record)
    _bust_list_cache()
    return {"ok": True, "child": child_id}


def complete(child_id, result, error=None, meta=None):
    """Record a worker result in the mailbox and flip status -> done|failed."""
    record = _find(child_id)
    if record is None:
        return {"ok": False, "error": f"no subagent {child_id}"}
    status = "failed" if error else "done"
    message = {
        "from": "worker",
        "type": "result" if not error else "error",
        "text": error if error else result,
        "ts": int(time.time() * 1000),
    }
    path = os.path.join(record["session_dir"], "mailbox.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(message) + "\n")
    record["status"] = status
    record["completed_ts"] = int(time.time() * 1000)
    if error:
        record["error"] = error
    if meta:
        record["meta"] = {**record.get("meta", {}), **meta}
    _write(record)
    _bust_list_cache()
    return {"ok": True, "status": status}


def prompt(child_id):
    record = _find(child_id)
    if record is None:
        return None
    path = os.path.join(record["session_dir"], "prompt.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def child_result(child_id):
    """Status + latest worker result for a child (None if not done yet)."""
    record = _find(child_id)
    if record is None:
        return {"ok": False, "error": f"no subagent {child_id}"}
    last = None
    path = os.path.join(record["session_dir"], "mailbox.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("from") == "worker":
                    last = msg
    return {
        "ok": True,
        "child": child_id,
        "status": record.get("status"),
        "result": last.get("text") if last else None,
        "error": last.get("type") == "error" if last else False,
    }


def list_subagents():
    now = int(time.time() * 1000)
    if _list_cache["data"] is not None and now - _list_cache["ts"] < _LIST_CACHE_TTL_MS:
        return _list_cache["data"]
    _ensure()
    out = []
    for entry in sorted(os.listdir(SUBAGENTS_DIR)):
        path = os.path.join(SUBAGENTS_DIR, entry, "admission.json")
        if os.path.exists(path):
            with open(path) as f:
                out.append(json.load(f))
    _list_cache["data"] = out
    _list_cache["ts"] = now
    return out


def mailbox(child_id):
    record = _find(child_id)
    if record is None:
        return {"ok": False, "error": f"no subagent {child_id}"}
    msgs = []
    path = os.path.join(record["session_dir"], "mailbox.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    msgs.append(json.loads(line))
    return {"ok": True, "messages": msgs}


def _find(child_id):
    path = os.path.join(SUBAGENTS_DIR, child_id, "admission.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write(record):
    import tempfile
    path = os.path.join(SUBAGENTS_DIR, record["rlm_child_id"], "admission.json")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
