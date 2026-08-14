"""Persistent IPython kernel sessions via jupyter_client.

Kernels are launched detached (start_new_session) and without JPY_PARENT_PID
so they survive the short-lived CLI process that started them. Each kernel
is stopped explicitly via `rlmlab kernel stop`.
"""

import json
import os
import signal
import subprocess
import time
import uuid

from jupyter_client import KernelManager

HOME = os.environ.get("RLMLAB_HOME", os.path.expanduser("~/.rlmlab"))
KERNELS_DIR = os.path.join(HOME, "kernels")
LOG_DIR = os.path.join(HOME, "logs")

KERNEL_SPEC = "python3"


def _ensure_dirs():
    os.makedirs(KERNELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def start(name):
    _ensure_dirs()
    if _read_index(name) is not None:
        return _error_result(f"kernel session {name!r} already exists (stop it first)")
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
        proc = subprocess.Popen(
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
    return session


def exec_code(name, code, timeout=120):
    _ensure_dirs()
    idx = _read_index(name)
    if idx is None:
        return _error_result(f"no kernel session named {name!r} (start one first)")
    km = KernelManager()
    km.load_connection_file(idx["connection_file"])
    client = km.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        msg_id = client.execute(code)
        result = _collect(client, msg_id, timeout)
        _touch(name)
        return result
    except Exception as e:
        return _error_result(f"kernel not responsive: {e}")
    finally:
        client.stop_channels()


def stop(name):
    idx = _read_index(name)
    if idx is None:
        return {"ok": False, "error": f"no kernel session named {name!r}"}
    try:
        os.kill(idx["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    _cleanup(name)
    return {"ok": True, "stopped": name}


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
        except Exception:
            return False
    finally:
        client.stop_channels()


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _write_index(name, session):
    with open(os.path.join(KERNELS_DIR, _safe(name) + ".json"), "w") as f:
        json.dump(session, f)


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
    outputs = []
    errors = []
    start = time.time()
    while True:
        if time.time() - start > timeout:
            errors.append(f"execution timed out after {timeout}s")
            break
        try:
            msg = client.get_iopub_msg(timeout=1)
        except Exception:
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
