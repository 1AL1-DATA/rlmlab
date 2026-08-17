# rlmlab #TODO test gates. content

A recursive language-machine harness (prime-agent style): persistent IPython
kernels + a continual harness + recursive subagents, driven by a local model
(llama.cpp / ollama). Adapters expose it to opencode and hermes.

## Mental model

- **Subagent / child** = one unit of work: a name, a prompt, a mailbox, a
  status (`admitted → running → done | failed`), and its own persistent
  IPython kernel that exists only while it runs.
- **Worker** = drains `admitted` children. Two executors:
  - `deterministic` — the prompt *is* Python code, run in the child's kernel.
  - `llm` — the prompt is a natural-language task; a local model drives the
    child's kernel (its only tool = run code) via a JSON action protocol.
- **Harness** = durable playbook (`prompt_notes`, `memories`,
  `skill_descriptions`, `subagent_specs`) with snapshot rollback. Injected as
  `NOTES` into every child kernel at boot. Workers auto-distill outcomes back
  into `memories` (the RL feedback loop).
- **Kernels survive crashes** — each kernel's user namespace is snapshotted
  with dill (per-variable, 256 MB cap) and restored on start. A stale
  dead-pid index entry is cleaned up and the session revived automatically,
  so a crash or reboot does not lose in-flight state.
- **Recursion** — a child can delegate to its own children (LLM: JSON actions
  `run_child` / `child_result` / `list_children`; deterministic: `run_child()`
  / `child_result()` helpers), bounded by depth (default 3). Parents block on
  `child_result`, draining pending descendants inline.

## How you interact

1. **CLI** — `rlmlab work once|supervise`, `rlm run|submit|mailbox`, `harness
   refine|rollback`, `kernel start|exec|reap`, `goals …`. Scriptable (cron /
   systemd).
2. **Task file** — `rlmlab rlm submit task.md` (`# title` first line, body is
   the prompt) admits a child.
3. **opencode / hermes** — the plugins expose the same operations as tools
   (`rlm_run`, `rlm_mailbox`, `harness_refine`, `kernel_exec`, …). You talk to
   the agent; it calls the harness.
4. **Ambient worker** — the systemd user timer
   (`rlmlab-work.timer`, every 5 min) or `rlmlab work supervise` drains
   admitted children without you doing anything.

Everything is persisted under `~/.rlmlab/` (`subagents/<id>/admission.json`,
`prompt.txt`, `mailbox.jsonl`; `harness/harness_state.json` + `snapshots/`).

## Typical flow

```
rlmlab rlm run --name stats --prompt "Compute the mean of [3,1,4,1,5,9] in Python."
  → admitted rlm_xxx
rlmlab work once --executor llm --base-url http://127.0.0.1:8080/v1 --model "<path>"
  → drained, result in mailbox, status done, memory distilled
rlmlab rlm mailbox --child rlm_xxx
```

## CLI reference

- `kernel start|exec|stop|list|reap --max-age-seconds N|snapshot|restore --session S`
- `harness get|refine --section --item|snapshots|rollback --version N`
- `goals create|list|done|drop`
- `rlm run|submit|send|list|mailbox`
- `work once [--limit N] [--executor deterministic|llm] [--model M]
  [--base-url U] [--max-turns N] [--max-seconds N]`
- `work supervise [--interval S] [--executor …] [--model …] …`
- `schema`

`--json` on any command gives strict JSON on stdout. Run `rlmlab --help` /
`rlmlab schema` for details.

## Serving the model

The LLM executor speaks the OpenAI chat-completions protocol. Target the
llama.cpp server (`:8080`, e.g. Qwen3.6-35B-A3B) or ollama (`:11434`). On an
8 GB laptop GPU with a 35B IQ2 model, expect ~7 tok/s; reduce the llama.cpp
`-c` context to reclaim VRAM/RAM for a significant speedup (hermes, for
example, refuses to start below a 64k context floor).

## Running as services

- **Model server** — a systemd user unit (`qwen.service` → `~/bin/qwen-start`)
  serves the llama.cpp model with `-c 65000`.
- **Ambient worker** — `rlmlab-work.timer` (every 5 min) drains admitted
  children, or `rlmlab work supervise` runs a foreground loop.
- **RLM gate test driver** — `rlm-gate.service` runs a checkpointed story
  transcript against a live hermes session to exercise the context-summary
  gates; see `/home/a/rlm-gate-story/README.md`.
