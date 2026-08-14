"""rlmlab CLI - single entry point, machine contracts on stdout.

Every command accepts --json for strict JSON output. `rlmlab schema`
prints the JSON schema for every tool.
"""

import argparse
import json
import sys

from . import __version__, apis, goals, harness, kernel, subagents, worker

PROTO_VERSION = "1"


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--version" in argv:
        print(f"rlmlab {__version__} (proto {PROTO_VERSION})")
        return 0

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit strict JSON on stdout")

    parser = argparse.ArgumentParser(prog="rlmlab", parents=[common])
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="tool", required=True)

    p = sub.add_parser("kernel", help="persistent IPython kernel sessions", parents=[common])
    ks = p.add_subparsers(dest="action", required=True)
    k_start = ks.add_parser("start", parents=[common])
    k_start.add_argument("--name", required=True)
    k_exec = ks.add_parser("exec", parents=[common])
    k_exec.add_argument("--session", required=True)
    k_exec.add_argument("--code", required=True)
    k_exec.add_argument("--timeout", type=int, default=120)
    k_stop = ks.add_parser("stop", parents=[common])
    k_stop.add_argument("--session", required=True)
    k_reap = ks.add_parser("reap", parents=[common])
    k_reap.add_argument("--max-age-seconds", type=int, default=3600)
    ks.add_parser("list", parents=[common])

    p = sub.add_parser("harness", help="continual harness state", parents=[common])
    hs = p.add_subparsers(dest="action", required=True)
    hs.add_parser("get", parents=[common])
    h_ref = hs.add_parser("refine", parents=[common])
    h_ref.add_argument("--section", required=True)
    h_ref.add_argument("--item", required=True)
    hs.add_parser("snapshots", parents=[common])
    h_rb = hs.add_parser("rollback", parents=[common])
    h_rb.add_argument("--version", type=int, required=True)

    p = sub.add_parser("goals", help="persistent goals", parents=[common])
    gs = p.add_subparsers(dest="action", required=True)
    g_create = gs.add_parser("create", parents=[common])
    g_create.add_argument("--text", required=True)
    g_list = gs.add_parser("list", parents=[common])
    g_list.add_argument("--status", default=None, choices=["open", "done", "dropped"])
    g_done = gs.add_parser("done", parents=[common])
    g_done.add_argument("--id", required=True)
    g_drop = gs.add_parser("drop", parents=[common])
    g_drop.add_argument("--id", required=True)

    p = sub.add_parser("rlm", help="recursive subagents (admission handles + mailboxes)", parents=[common])
    rs = p.add_subparsers(dest="action", required=True)
    r_run = rs.add_parser("run", parents=[common])
    r_run.add_argument("--name", required=True)
    r_run.add_argument("--prompt", required=True)
    r_run.add_argument("--api", default=None, help="api key (default: RLMLAB_DEFAULT_API or deterministic)")
    r_send = rs.add_parser("send", parents=[common])
    r_send.add_argument("--child", required=True)
    r_send.add_argument("--text", required=True)
    r_submit = rs.add_parser("submit", parents=[common])
    r_submit.add_argument("--api", default=None)
    r_submit.add_argument("file")
    r_setapi = rs.add_parser("set-api", parents=[common])
    r_setapi.add_argument("--child", required=True)
    r_setapi.add_argument("--api", required=True)
    rs.add_parser("list", parents=[common])
    r_mail = rs.add_parser("mailbox", parents=[common])
    r_mail.add_argument("--child", required=True)

    sub.add_parser("schema", parents=[common])

    p = sub.add_parser("apis", help="named executor/API registry", parents=[common])
    a_s = p.add_subparsers(dest="action", required=True)
    a_s.add_parser("list", parents=[common])
    a_reg = a_s.add_parser("register", parents=[common])
    a_reg.add_argument("--name", required=True)
    a_reg.add_argument("--type", choices=["deterministic", "llama", "ollama"], required=True)
    a_reg.add_argument("--params", default="{}", help="JSON object with base_url/model")
    a_dereg = a_s.add_parser("deregister", parents=[common])
    a_dereg.add_argument("--name", required=True)

    p = sub.add_parser("work", help="drain admitted subagents", parents=[common])
    w = p.add_subparsers(dest="action", required=True)
    w_once = w.add_parser("once", parents=[common])
    w_once.add_argument("--limit", type=int, default=None)
    w_once.add_argument("--timeout", type=int, default=120)
    w_once.add_argument("--executor", choices=["deterministic", "llm"], default="deterministic")
    w_once.add_argument("--api", default=None, help="only drain tasks assigned to this api key")
    w_once.add_argument("--model", default=None)
    w_once.add_argument("--base-url", default=None)
    w_once.add_argument("--max-turns", type=int, default=None)
    w_once.add_argument("--max-seconds", type=int, default=None)
    w_sup = w.add_parser("supervise", parents=[common])
    w_sup.add_argument("--interval", type=int, default=5)
    w_sup.add_argument("--timeout", type=int, default=120)
    w_sup.add_argument("--executor", choices=["deterministic", "llm"], default="llm")
    w_sup.add_argument("--api", default=None, help="only drain tasks assigned to this api key")
    w_sup.add_argument("--model", default=None)
    w_sup.add_argument("--base-url", default=None)
    w_sup.add_argument("--max-turns", type=int, default=None)
    w_sup.add_argument("--max-seconds", type=int, default=None)

    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(f"rlmlab {__version__} (proto {PROTO_VERSION})")
        return 0

    result = dispatch(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(args.tool, getattr(args, "action", None), result)
    return 0 if isinstance(result, dict) and result.get("ok", True) else 1


def dispatch(args):
    tool = args.tool
    if tool == "schema":
        return {"proto_version": PROTO_VERSION, "schema": build_schema()}
    if tool == "kernel":
        return dispatch_kernel(args)
    if tool == "harness":
        return dispatch_harness(args)
    if tool == "goals":
        return dispatch_goals(args)
    if tool == "rlm":
        return dispatch_rlm(args)
    if tool == "apis":
        return dispatch_apis(args)
    if tool == "work":
        return dispatch_work(args)
    return {"ok": False, "error": f"unknown tool {tool}"}


def dispatch_kernel(args):
    if args.action == "start":
        return {"ok": True, "session": kernel.start(args.name)}
    if args.action == "exec":
        return kernel.exec_code(args.session, args.code, timeout=args.timeout)
    if args.action == "stop":
        return kernel.stop(args.session)
    if args.action == "reap":
        return worker.reap(max_age_seconds=args.max_age_seconds)
    if args.action == "list":
        return {"ok": True, "sessions": kernel.list_sessions()}
    return {"ok": False, "error": "unknown kernel action"}


def dispatch_harness(args):
    if args.action == "get":
        return {"ok": True, "state": harness.get_state()}
    if args.action == "refine":
        return harness.refine(args.section, args.item)
    if args.action == "snapshots":
        return {"ok": True, "snapshots": harness.list_snapshots()}
    if args.action == "rollback":
        return harness.rollback(args.version)
    return {"ok": False, "error": "unknown harness action"}


def dispatch_goals(args):
    if args.action == "create":
        return goals.create(args.text)
    if args.action == "list":
        return {"ok": True, "goals": goals.list_goals(args.status)}
    if args.action == "done":
        return goals.done(args.id)
    if args.action == "drop":
        return goals.drop(args.id)
    return {"ok": False, "error": "unknown goals action"}


def dispatch_rlm(args):
    if args.action == "run":
        return subagents.run(args.name, args.prompt, api=args.api)
    if args.action == "send":
        return subagents.send(args.child, args.text)
    if args.action == "submit":
        return _submit_file(args.file, api=args.api)
    if args.action == "set-api":
        return subagents.set_api(args.child, args.api)
    if args.action == "list":
        return {"ok": True, "subagents": subagents.list_subagents()}
    if args.action == "mailbox":
        return subagents.mailbox(args.child)
    return {"ok": False, "error": "unknown rlm action"}


def _submit_file(path, api=None):
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        return {"ok": False, "error": f"cannot read {path}: {e}"}
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return {"ok": False, "error": "empty task file"}
    name = lines[0].strip().lstrip("# ").strip()[:64]
    prompt = "\n".join(lines[1:]).strip()
    if not prompt:
        return {"ok": False, "error": "task file has a title but no prompt body"}
    return subagents.run(name, prompt, api=api)


def dispatch_apis(args):
    if args.action == "list":
        return apis.list_apis()
    if args.action == "register":
        try:
            params = json.loads(args.params) if args.params else {}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"--params must be JSON: {e}"}
        return apis.register(args.name, args.type, params)
    if args.action == "deregister":
        return apis.deregister(args.name)
    return {"ok": False, "error": "unknown apis action"}


def dispatch_work(args):
    llm_opts = {}
    for key in ("model", "base_url", "max_turns", "max_seconds"):
        val = getattr(args, key, None)
        if val:
            llm_opts[key] = val
    if args.action == "once":
        return worker.work_once(
            limit=args.limit,
            timeout=args.timeout,
            executor=args.executor,
            llm_opts=llm_opts,
            api=args.api,
        )
    if args.action == "supervise":
        return worker.supervise(
            interval=args.interval,
            timeout=args.timeout,
            executor=args.executor,
            llm_opts=llm_opts,
            api=args.api,
        )
    return {"ok": False, "error": "unknown work action"}


def build_schema():
    return {
        "kernel.start": {"args": {"name": "string"}},
        "kernel.exec": {"args": {"session": "string", "code": "string", "timeout": "int?"}},
        "kernel.stop": {"args": {"session": "string"}},
        "kernel.reap": {"args": {"max_age_seconds": "int?"}},
        "kernel.list": {"args": {}},
        "harness.get": {"args": {}},
        "harness.refine": {"args": {"section": "string", "item": "string"}},
        "harness.snapshots": {"args": {}},
        "harness.rollback": {"args": {"version": "int"}},
        "goals.create": {"args": {"text": "string"}},
        "goals.list": {"args": {"status": "string?"}},
        "goals.done": {"args": {"id": "string"}},
        "goals.drop": {"args": {"id": "string"}},
        "rlm.run": {"args": {"name": "string", "prompt": "string", "api": "string?"}},
        "rlm.send": {"args": {"child": "string", "text": "string"}},
        "rlm.submit": {"args": {"file": "string", "api": "string?"}},
        "rlm.set-api": {"args": {"child": "string", "api": "string"}},
        "rlm.list": {"args": {}},
        "rlm.mailbox": {"args": {"child": "string"}},
        "apis.list": {"args": {}},
        "apis.register": {"args": {"name": "string", "type": "deterministic|llama|ollama", "params": "json?"}},
        "apis.deregister": {"args": {"name": "string"}},
        "work.once": {"args": {"limit": "int?", "timeout": "int?", "executor": "deterministic|llm",
                               "api": "string?", "model": "string?", "base_url": "string?",
                               "max_turns": "int?", "max_seconds": "int?"}},
        "work.supervise": {"args": {"interval": "int?", "timeout": "int?", "executor": "deterministic|llm",
                                    "api": "string?", "model": "string?", "base_url": "string?",
                                    "max_turns": "int?", "max_seconds": "int?"}},
    }


def print_human(tool, action, result):
    if isinstance(result, dict) and result.get("ok") is False:
        print(f"ERROR: {result.get('error')}", file=sys.stderr)
        return
    if tool == "kernel":
        if action == "exec":
            for o in result.get("outputs", []):
                print(o.get("text", ""), end="")
            if result.get("outputs"):
                print()
        elif action == "start":
            s = result["session"]
            print(f"kernel {s['name']} started (kernel_id={s['kernel_id']})")
        elif action == "stop":
            print(f"stopped {result['stopped']}")
        elif action == "list":
            for s in result["sessions"]:
                print(f"{s['name']}  last_active={s.get('last_active')}")
    elif tool == "harness":
        if action == "get":
            print(json.dumps(result["state"], indent=2))
        elif action == "refine":
            print(f"harness v{result['version']} ({result['section']}: {result['count']} items)")
        elif action == "snapshots":
            print("\n".join(result["snapshots"]) or "no snapshots")
        elif action == "rollback":
            print(f"rolled back to v{result['version']}")
    elif tool == "goals":
        if action == "create":
            print(f"created {result['goal']['id']}: {result['goal']['text']}")
        elif action == "list":
            for g in result["goals"]:
                print(f"{g['id']} [{g['status']}] {g['text']}")
        elif action in ("done", "drop"):
            print(f"{result['id']} -> {result['status']}")
    elif tool == "rlm":
        if action in ("run", "submit"):
            print(f"admitted {result['rlm_child_id']} ({result['name']}) api={result.get('api','deterministic')} -> {result['session_dir']}")
        elif action == "send":
            print(f"message queued to {result['message']['from']}")
        elif action == "set-api":
            print(f"{result['child']} -> api {result['api']}")
        elif action == "list":
            for s in result["subagents"]:
                print(f"{s['rlm_child_id']} [{s['status']}] api={s.get('api','deterministic')} {s['name']}")
        elif action == "mailbox":
            for m in result["messages"]:
                print(f"[{m['from']}] {m['text']}")
    elif tool == "apis":
        if action == "list":
            for a in result["apis"]:
                mark = "up" if a.get("online") else "down"
                print(f"{a['name']}  [{a['type']}]  {mark}")
        elif action == "register":
            print(f"registered {result['name']} ({result['config']['type']})")
        elif action == "deregister":
            print(f"deregistered {result['name']}")
    elif tool == "work":
        if action == "once":
            for r in result["processed"]:
                if r.get("ok"):
                    print(f"  ok   {r['child']}: {r.get('text','')[:80]}")
                else:
                    print(f"  FAIL {r.get('child')}: {r.get('error','')[:80]}")
            print(f"processed {len(result['processed'])} subagent(s)")
    elif tool == "schema":
        print(json.dumps(result["schema"], indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
