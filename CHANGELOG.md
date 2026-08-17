# Changelog

All notable changes to rlmlab. Unreleased changes describe work on `main`
not yet tagged.

## Unreleased

### Kernels
- Dill-based snapshot/restore: `kernel snapshot --session S` and
  `kernel restore --session S`. Each kernel snapshots its user namespace
  (per-variable, 256 MB cap, manifest JSON) on stop/start so a crash or
  reboot revives in-flight state.
- Stale-index recovery: starting a session whose pid is dead cleans up the
  stale entry and restarts it (with snapshot restore) instead of erroring.

### Persistence hardening
- Harness, goals, and apis writes are now atomic (temp file + `os.replace`),
  so a kill mid-write cannot corrupt state.
- Harness snapshots are taken every 10 refinements (was every call).

### Subagents / worker
- `list_subagents` is memoised with a 5 s TTL, busted on `run`/`claim`/
  `complete` — faster supervision loops.
- Memory distillation failure no longer fails the subagent; only expected
  write errors are swallowed.

### Security
- urllib call sites (LLM executor + API probe) are restricted to http/https
  schemes.
- Broad exception handlers scoped to their intent (kernel poll timeouts via
  `queue.Empty`; best-effort memory writes via `OSError`/`ValueError`).

### Misc
- `_extract_json` also parses array-shaped LLM responses.
- Added `dill` and `ruff` (dev) dependencies.