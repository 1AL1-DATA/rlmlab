"""Persistent IPython kernel sessions via jupyter_client.

Kernels are launched detached (start_new_session) and without JPY_PARENT_PID
so they survive the short-lived CLI process that started them. Each kernel
is stopped explicitly via `rlmlab kernel stop`.
"""

import json
import os
import queue
import signal
import subprocess  # nosec B404 - only used to launch the ipykernel child (shell=False, static argv)
import time
import uuid

from jupyter_client import KernelManager

HOME = os.environ.get("RLMLAB_HOME", os.path.expanduser("~/.rlmlab"))
KERNELS_DIR = os.path.join(HOME, "kernels")
LOG_DIR = os.path.join(HOME, "logs")

KERNEL_SPEC = "python3"

SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024
SNAPSHOT_MARKER = "__RLMLAB_SNAP__"
RESTORE_MARKER = "__RLMLAB_RESTORE__"


def _state_path(name):
    return os.path.join(KERNELS_DIR, _safe(name) + ".state.dill")


def _manifest_path(name):
    return os.path.join(KERNELS_DIR, _safe(name) + ".state.manifest.json")


def _ensure_dirs():
    os.makedirs(KERNELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def start(name):
    _ensure_dirs()
    existing = _read_index(name)
    if existing is not None:
        if _pid_alive(existing):
            return _error_result(f"kernel session {name!r} already exists (stop it first)")
        _cleanup(name)  # stale dead-pid index; fresh start restores the dill snapshot
    km = KernelManager(kernel_name=KERNEL_SPEC)
    km.write_connection_file()
    stable_conn = os.path.join(KERNELS_DIR, _safe(name) + ".connection.json")
    with open(km.connection_file) as f:
        content = f.read()
    with open(stable_conn, "w") as f:
        f.write(content)
    cmd = [stable_conn if a == km.connection_file else a for a in km.format_kernel_cmd()]
    env = dict(os.environ)
    env.pop("JPY_PARENT_PID", None)
    log_path = os.path.join(LOG_DIR, _safe(name) + ".log")
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(  # nosec B603 - argv from KernelManager, shell=False, no user input
            cmd,
            env=env,
            start_new_session=True,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
    deadline = time.time() + 60
    while time.time() < deadline:
        if _kernel_ready(stable_conn):
            break
        if proc.poll() is not None:
            proc.wait()
            return _error_result(f"kernel exited at startup (see {log_path})")
        time.sleep(0.2)
    else:
        proc.kill()
        return _error_result("kernel did not become ready in 60s")
    session = {
        "name": name,
        "kernel_id": str(uuid.uuid4()),
        "pid": proc.pid,
        "connection_file": stable_conn,
        "created": int(time.time() * 1000),
        "last_active": int(time.time() * 1000),
    }
    _write_index(name, session)
    restored = restore(name)
    session["restored"] = restored
    _write_index(name, session)
    return session


def _snapshot_code(out_path, manifest_path, max_bytes):
    """Python run inside the kernel: dill-serialize user_ns per-variable,
    skipping unpicklable names, to out_path + a JSON manifest."""
    return f"""
import builtins as _b, json, os
try:
    import dill
except Exception as _err:
    _b.print("{SNAPSHOT_MARKER}" + json.dumps({{"error": "dill unavailable: " + _b.str(_err)}}))
else:
    dill.settings["recurse"] = True
    _ip = get_ipython()
    _ns = _ip.user_ns if _ip is not None else _b.globals()
    _hidden = _b.set(_b.getattr(_ip, "user_ns_hidden", {{}}) or {{}}) if _ip is not None else _b.set()
    _always_skip = {{"rlm", "asyncio", "In", "Out", "get_ipython", "exit", "quit", "open", "_"}}
    _payload = {{}}
    _skipped = []
    _total = 0
    for _name, _val in list(_ns.items()):
        if _name in _always_skip or _name in _hidden or _name.startswith("_"):
            continue
        try:
            _blob = dill.dumps(_val)
        except Exception as _e:
            _skipped.append({{"name": _name, "reason": _b.str(_e)[:200]}})
            continue
        if _total + _b.len(_blob) > {max_bytes}:
            _skipped.append({{"name": _name, "reason": "over max bytes"}})
            continue
        _payload[_name] = _blob
        _total += _b.len(_blob)
    try:
        import tempfile
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname({out_path!r}))
        with os.fdopen(_fd, "wb") as _f:
            dill.dump(_payload, _f)
        os.replace(_tmp, {out_path!r})
    except Exception as _e:
        _b.print("{SNAPSHOT_MARKER}" + json.dumps({{"error": _b.str(_e)}}))
    else:
        with open({manifest_path!r}, "w") as _f:
            json.dump({{"saved": _b.sorted(_payload), "skipped": _skipped, "bytes": _total}}, _f)
        _b.print("{SNAPSHOT_MARKER}" + json.dumps({{"saved": _b.len(_payload), "skipped": _b.len(_skipped), "bytes": _total}}))
"""


def _restore_code(in_path):
    """Python run inside the kernel: revive user_ns from a dill snapshot."""
    return f"""
import builtins as _b, json
try:
    import dill
except Exception as _err:
    _b.print("{RESTORE_MARKER}" + json.dumps({{"error": "dill unavailable: " + _b.str(_err)}}))
else:
    dill.settings["recurse"] = True
    _ip = get_ipython()
    _ns = _ip.user_ns if _ip is not None else _b.globals()
    try:
        with open({in_path!r}, "rb") as _f:
            _payload = dill.load(_f)
    except Exception as _e:
        _b.print("{RESTORE_MARKER}" + json.dumps({{"error": "load failed: " + _b.str(_e)}}))
    else:
        _restored = []
        _failed = []
        for _name, _blob in _payload.items():
            try:
                _ns[_name] = dill.loads(_blob)
                _restored.append(_name)
            except Exception as _e:
                _failed.append({{"name": _name, "reason": _b.str(_e)[:200]}})
        _b.print("{RESTORE_MARKER}" + json.dumps({{"restored": _restored, "failed": _failed}}))
"""


def snapshot(name, timeout=120):
    """Best-effort dill snapshot of a kernel's user namespace."""
    idx = _read_index(name)
    if idx is None:
        return _error_result(f"no kernel session named {name!r}")
    result = _run_raw(idx, _snapshot_code(_state_path(name), _manifest_path(name), SNAPSHOT_MAX_BYTES), timeout)
    if not result.get("ok"):
        return result
    payload = _parse_marker(result, SNAPSHOT_MARKER)
    if payload is None:
        return _error_result("snapshot marker not found in kernel output")
    if payload.get("error"):
        return _error_result(payload["error"])
    payload["ok"] = True
    payload["path"] = _state_path(name)
    return payload


def restore(name, timeout=120):
    """Restore a kernel's user namespace from its snapshot (if any)."""
    idx = _read_index(name)
    if idx is None:
        return _error_result(f"no kernel session named {name!r}")
    path = _state_path(name)
    if not os.path.exists(path):
        return {"ok": True, "restored": [], "note": "no snapshot"}
    result = _run_raw(idx, _restore_code(path), timeout)
    if not result.get("ok"):
        return result
    payload = _parse_marker(result, RESTORE_MARKER)
    if payload is None:
        return _error_result("restore marker not found in kernel output")
    if payload.get("error"):
        return _error_result(payload["error"])
    payload["path"] = path
    return payload


def _parse_marker(result, marker):
    text = result.get("text", "") or ""
    idx = text.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    return json.loads(text[start:start + 4096].strip())


def _run_raw(idx, code, timeout):
    """Execute code against an existing kernel index without touching last_active."""
    km = KernelManager()
    km.load_connection_file(idx["connection_file"])
    client = km.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        msg_id = client.execute(code)
        return _collect(client, msg_id, timeout)
    except Exception as e:  # noqa: BLE001 - surface any kernel transport failure as an error result
        return _error_result(f"kernel not responsive: {e}")
    finally:
        client.stop_channels()


def exec_code(name, code, timeout=120):
    _ensure_dirs()
    idx = _read_index(name)
    if idx is None:
        return _error_result(f"no kernel session named {name!r} (start one first)")
    result = _run_raw(idx, code, timeout)
    if result.get("ok"):
        _touch(name)
    return result


def stop(name, snapshot_state=True):
    idx = _read_index(name)
    if idx is None:
        return {"ok": False, "error": f"no kernel session named {name!r}"}
    snap = None
    if snapshot_state and _pid_alive(idx):
        snap = snapshot(name)
    try:
        os.kill(idx["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    _cleanup(name)
    out = {"ok": True, "stopped": name}
    if snap is not None and snap.get("ok") is False:
        out["snapshot_error"] = snap.get("error")
    return out


def list_sessions():
    _ensure_dirs()
    out = []
    for fn in sorted(os.listdir(KERNELS_DIR)):
        if not fn.endswith(".json"):
            continue
        idx = _read_index(fn[:-5])
        if idx:
            idx["alive"] = _pid_alive(idx)
            out.append(idx)
    return out


def _pid_alive(idx):
    pid = idx.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kernel_ready(connection_file):
    probe = KernelManager()
    probe.load_connection_file(connection_file)
    client = probe.client()
    client.start_channels()
    try:
        try:
            client.wait_for_ready(timeout=5)
            return True
        except Exception:  # noqa: BLE001 - probe returns False on any failure
            return False
    finally:
        client.stop_channels()


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _write_index(name, session):
    import tempfile
    path = os.path.join(KERNELS_DIR, _safe(name) + ".json")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(session, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _read_index(name):
    path = os.path.join(KERNELS_DIR, _safe(name) + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _cleanup(name):
    path = os.path.join(KERNELS_DIR, _safe(name) + ".json")
    if os.path.exists(path):
        with open(path) as f:
            idx = json.load(f)
        os.remove(path)
        try:
            os.remove(idx["connection_file"])
        except OSError:
            pass


def _touch(name):
    idx = _read_index(name)
    if idx:
        idx["last_active"] = int(time.time() * 1000)
        _write_index(name, idx)


def _collect(client, msg_id, timeout):
    """Collect IOPub messages until idle status or timeout.
    Uses short polling intervals to reduce CPU churn.
    """
    outputs = []
    errors = []
    start = time.time()
    poll_interval = 0.1  # ponytail: use 0.1s poll; switch to select() if CPU matters
    while True:
        if time.time() - start > timeout:
            errors.append(f"execution timed out after {timeout}s")
            break
        try:
            msg = client.get_iopub_msg(timeout=poll_interval)
        except queue.Empty:  # timeout is the normal poll path; keep polling until deadline
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        content = msg["content"]
        if mtype == "stream":
            outputs.append({"type": "stream", "name": content["name"], "text": content["text"]})
        elif mtype == "execute_result":
            outputs.append({"type": "result", "text": content.get("data", {}).get("text/plain", "")})
        elif mtype == "display_data":
            data = content.get("data", {})
            if "text/plain" in data:
                outputs.append({"type": "display", "text": data["text/plain"]})
        elif mtype == "error":
            errors.append("\n".join(content.get("traceback", [])) or content.get("evalue", "error"))
        elif mtype == "status" and content.get("execution_state") == "idle":
            break
    return _error_result("\n".join(errors)) if errors else {
        "ok": True,
        "outputs": outputs,
        "text": "".join(o["text"] for o in outputs if o.get("text")),
    }


def _error_result(message):
    return {"ok": False, "error": message}
