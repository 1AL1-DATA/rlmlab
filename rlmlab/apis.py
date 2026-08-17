"""RLM API registry: named executor configs that admitted tasks can be
assigned to.

Each registered "API" is a named executor configuration with a type and
type-specific parameters:

    deterministic  -- run the prompt as Python code (no model)
    llama          -- llama.cpp OpenAI-compatible server (base_url, model)
    ollama         -- ollama OpenAI-compatible server (base_url, model)

The registry is persisted at ~/.rlmlab/apis.json. The `deterministic` entry
is always present; llama/ollama endpoints are auto-discovered by probing
their health/tags endpoints on every list, so the dashboard always shows a
live view of which APIs are reachable.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

HOME = os.environ.get("RLMLAB_HOME", os.path.expanduser("~/.rlmlab"))
APIS_FILE = os.path.join(HOME, "apis.json")

VALID_TYPES = ("deterministic", "llama", "ollama")

DETERMINISTIC = "deterministic"
DEFAULT_LLAMA = {"type": "llama", "base_url": "http://127.0.0.1:8080"}
DEFAULT_OLLAMA = {"type": "ollama", "base_url": "http://127.0.0.1:11434"}

EMPTY = {"version": 0, "apis": {}}


def _ensure():
    os.makedirs(HOME, exist_ok=True)
    if not os.path.exists(APIS_FILE):
        with open(APIS_FILE, "w") as f:
            json.dump(EMPTY, f, indent=2)


def _load():
    _ensure()
    with open(APIS_FILE) as f:
        return json.load(f)


def _save(state):
    import tempfile
    os.makedirs(HOME, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=HOME)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, APIS_FILE)
    except BaseException:
        os.unlink(tmp)
        raise


def _probe(base_url, path="", timeout=2):
    """True if the endpoint answers a lightweight HTTP request."""
    if urllib.parse.urlparse(base_url).scheme not in ("http", "https"):
        return False
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:  # nosec B310 - scheme validated above (http/https only)
            return resp.status < 400
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


_PROBE_PATHS = {
    "llama": "/health",
    "ollama": "/api/tags",
}


_DISCOVER_CACHE = {"ts": 0, "found": {}}
_DISCOVER_TTL_MS = 30_000


def auto_discover():
    """Probe llama + ollama defaults and scan common ports for live endpoints.
    Returns {name: config} for all discovered instances.

    The probe sweep is sequential and slow (defaults + up to ~20 ports x 2
    endpoints); it runs inside every ``resolve``/``get_api`` call, which the
    worker hits once per task — including deterministic tasks that never
    touch a model. The result is cached for a short TTL so the hot path
    pays for discovery at most once per 30 s regardless of task volume.
    """
    now = time.time() * 1000
    if _DISCOVER_CACHE["found"] and now - _DISCOVER_CACHE["ts"] < _DISCOVER_TTL_MS:
        return _DISCOVER_CACHE["found"]
    found = {}
    # Defaults
    for api_type, cfg in (("llama", DEFAULT_LLAMA), ("ollama", DEFAULT_OLLAMA)):
        if _probe(cfg["base_url"], _PROBE_PATHS[api_type]):
            name = "llama-8080" if api_type == "llama" else "ollama"
            found[name] = {"type": api_type, **{k: v for k, v in cfg.items() if k != "type"}}
    # Scan common ports (ponytail: 8080+11434 are already in defaults, scan 8000-8100)
    for port in range(8000, 8100, 10):
        url = f"http://127.0.0.1:{port}"
        for api_type, path in (("llama", "/health"), ("ollama", "/api/tags")):
            if _probe(url, path):
                name = f"{api_type}-{port}"
                found[name] = {"type": api_type, "base_url": url}
    _DISCOVER_CACHE["found"] = found
    _DISCOVER_CACHE["ts"] = now
    return found


def list_apis(refresh=True):
    """Merged registry view: saved configs + always-on deterministic + live
    discovered endpoints."""
    state = _load()
    apis = dict(state.get("apis", {}))
    apis.setdefault(DETERMINISTIC, {"type": "deterministic"})
    if refresh:
        for name, cfg in auto_discover().items():
            if name not in apis:
                apis[name] = cfg
    out = []
    for name, cfg in sorted(apis.items()):
        entry = {"name": name, **cfg}
        entry["online"] = _api_online(entry)
        out.append(entry)
    return {"ok": True, "apis": out}


def get_api(name):
    state = _load()
    apis = dict(state.get("apis", {}))
    apis.setdefault(DETERMINISTIC, {"type": "deterministic"})
    for discovered, cfg in auto_discover().items():
        apis.setdefault(discovered, cfg)
    return apis.get(name)


def register(name, api_type, params=None):
    if api_type not in VALID_TYPES:
        return {"ok": False, "error": f"invalid type {api_type!r}; choose from {', '.join(VALID_TYPES)}"}
    if not name or not str(name).strip():
        return {"ok": False, "error": "api name must be non-empty"}
    cfg = {"type": api_type, **(params or {})}
    if api_type == "deterministic":
        cfg = {"type": "deterministic"}
    state = _load()
    if state.get("apis", {}).get(name) is not None and state["apis"][name] == cfg:
        return {"ok": True, "name": name, "config": cfg}
    state.setdefault("apis", {})[name] = cfg
    state["version"] += 1
    _save(state)
    return {"ok": True, "name": name, "config": cfg}


def deregister(name):
    if name == DETERMINISTIC:
        return {"ok": False, "error": "the deterministic executor cannot be deregistered"}
    state = _load()
    if name not in state.get("apis", {}):
        return {"ok": False, "error": f"no registered api {name!r}"}
    del state["apis"][name]
    state["version"] += 1
    _save(state)
    return {"ok": True, "name": name}


def resolve(name):
    """Normalize an api key into executor params used by the worker:
    {executor, base_url, model}."""
    cfg = get_api(name)
    if cfg is None:
        return None
    api_type = cfg.get("type")
    if api_type == "deterministic":
        return {"executor": "deterministic"}
    executor = "llm"
    if api_type == "llama":
        base_url = cfg.get("base_url", DEFAULT_LLAMA["base_url"])
        model = cfg.get("model") or ""
    elif api_type == "ollama":
        base_url = cfg.get("base_url", DEFAULT_OLLAMA["base_url"])
        model = cfg.get("model") or ""
    else:
        return {"executor": "deterministic"}
    return {"executor": executor, "base_url": base_url, "model": model}


def _api_online(entry):
    api_type = entry.get("type")
    if api_type == "deterministic":
        return True
    if api_type in ("llama", "ollama"):
        return _probe(entry.get("base_url", ""), _PROBE_PATHS.get(api_type, ""))
    return False
