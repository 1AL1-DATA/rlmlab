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

## Unreleased (2nd round — review fixes)

### Correctness
- **Stuck-subagent fix**: `_process_one` now marks a child `failed` on every
  exit path — `kernel.start()` failure, failed note injection, failed llm
  exec, failed code exec, or any unexpected exception. Previously a
  `kernel.start()` failure or an exception left the child `running` forever.
- **Stale-running recovery**: `work_once` requeues children stuck in
  `running` whose claim is older than 10 min and whose kernel is dead
  (`_requeue_stale_running`), so a worker killed mid-run (kill -9, crash)
  no longer strands a child permanently. New `subagents.requeue`.
- **Atomic claim**: `subagents.claim` now takes a per-child flock around the
  read-modify-write, so two racing workers (or worker + CLI) can never
  double-claim a child.
- **Harness write lock**: `harness.refine` takes a flock around the
  read-modify-write; concurrent writers no longer lose updates.
- **API discovery hot path**: `apis.auto_discover` caches probe results for
  30 s. Previously the sequential port sweep (defaults + ~20 ports x 2
  probes) ran on every `resolve` — i.e. once per task, even deterministic
  ones that never touch a model.
- **Retry on model calls**: `agent_loop._complete` retries connection-level
  failures and 5xx responses twice with backoff before giving up; 4xx
  (config errors) fail fast.

### Features
- **Goals wired in**: open goals are injected into every child's preamble
  (`NOTES["goals"]`), and `rlmlab goals work --id G` admits a subagent from
  a goal's text and marks it done.
- **Scoped memory retrieval**: `_distill_notes` now pulls top-k memories by
  keyword overlap with the task prompt, capped at an 8 KB budget, instead of
  dumping up to 500 memories into every child.
- **Usage telemetry persisted**: llm executor stores `turns`/`tokens` in the
  child record (`meta`), surfaced by `rlmlab rlm list`.

### CI / docs
- Added `.github/workflows/ci.yml` (ruff + pytest on push/PR).
- README no longer claims opencode/hermes plugins ship in this repo (they
  don't); points to the CLI's `--format json` instead.