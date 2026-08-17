"""Persistent goals, stored as an append-only JSONL ledger."""

import json
import os
import time
import uuid

HOME = os.environ.get("RLMLAB_HOME", os.path.expanduser("~/.rlmlab"))
GOALS_FILE = os.path.join(HOME, "goals.jsonl")


def _ensure():
    os.makedirs(HOME, exist_ok=True)


def create(text):
    _ensure()
    goal = {
        "id": "goal_" + uuid.uuid4().hex[:16],
        "text": text,
        "status": "open",
        "created": int(time.time() * 1000),
    }
    with open(GOALS_FILE, "a") as f:
        f.write(json.dumps(goal) + "\n")
    return {"ok": True, "goal": goal}


def list_goals(status=None):
    _ensure()
    out = []
    if not os.path.exists(GOALS_FILE):
        return out
    with open(GOALS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            goal = json.loads(line)
            if status is None or goal["status"] == status:
                out.append(goal)
    return out


def done(goal_id):
    _ensure()
    return _set_status(goal_id, "done")


def drop(goal_id):
    _ensure()
    return _set_status(goal_id, "dropped")


def _set_status(goal_id, status):
    if not os.path.exists(GOALS_FILE):
        return {"ok": False, "error": f"no goal {goal_id}"}
    import tempfile
    updated = []
    found = False
    with open(GOALS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            goal = json.loads(line)
            if goal["id"] == goal_id:
                goal["status"] = status
                goal["updated"] = int(time.time() * 1000)
                found = True
            updated.append(goal)
    if not found:
        return {"ok": False, "error": f"no goal {goal_id}"}
    # Atomic write: temp file + os.replace to prevent corruption on kill
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(GOALS_FILE))
    try:
        with os.fdopen(fd, "w") as f:
            for g in updated:
                f.write(json.dumps(g) + "\n")
        os.replace(tmp, GOALS_FILE)
    except BaseException:
        os.unlink(tmp)
        raise
    return {"ok": True, "id": goal_id, "status": status}
