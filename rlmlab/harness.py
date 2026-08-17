"""Continual harness state: durable playbook with snapshot-based rollback."""

import json
import os
import time

HOME = os.environ.get("RLMLAB_HOME", os.path.expanduser("~/.rlmlab"))
HARNESS_DIR = os.path.join(HOME, "harness")
STATE_FILE = os.path.join(HARNESS_DIR, "harness_state.json")
SNAPSHOTS_DIR = os.path.join(HARNESS_DIR, "snapshots")

EMPTY_STATE = {
    "version": 0,
    "prompt_notes": [],
    "memories": [],
    "skill_descriptions": [],
    "subagent_specs": [],
}

MEMORY_CAP = 500
SNAPSHOT_INTERVAL = 10  # only snapshot every N refinements


def _ensure():
    os.makedirs(HARNESS_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump(EMPTY_STATE, f, indent=2)


def get_state():
    _ensure()
    with open(STATE_FILE) as f:
        return json.load(f)


def refine(section, item):
    _ensure()
    state = get_state()
    if section not in EMPTY_STATE:
        return {"ok": False, "error": f"unknown section {section!r}"}
    state[section].append({"text": item, "added": int(time.time() * 1000)})
    if section == "memories" and len(state["memories"]) > MEMORY_CAP:
        state["memories"] = state["memories"][-MEMORY_CAP:]
    state["version"] += 1
    _write(state)
    # ponytail: snapshot only every SNAPSHOT_INTERVAL refinements, not every call
    if state["version"] % SNAPSHOT_INTERVAL == 1:
        _snapshot({k: list(v) if isinstance(v, list) else v for k, v in state.items()})
    return {"ok": True, "version": state["version"], "section": section, "count": len(state[section])}


def list_snapshots():
    _ensure()
    out = []
    for fn in sorted(os.listdir(SNAPSHOTS_DIR)):
        if fn.endswith(".json"):
            out.append(fn[:-5])
    return out


def rollback(version):
    _ensure()
    path = os.path.join(SNAPSHOTS_DIR, f"v{version}.json")
    if not os.path.exists(path):
        return {"ok": False, "error": f"no snapshot for version {version}"}
    with open(path) as f:
        state = json.load(f)
    _write(state)
    return {"ok": True, "version": state["version"]}


def _snapshot(state):
    with open(os.path.join(SNAPSHOTS_DIR, f"v{state['version']}.json"), "w") as f:
        json.dump(state, f, indent=2)


def _write(state):
    # Atomic write: temp file + os.replace to prevent corruption on kill
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except BaseException:
        os.unlink(tmp)
        raise
