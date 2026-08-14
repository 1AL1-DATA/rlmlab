import json

from rlmlab import subagents, worker


def test_wrangler_preamble_gates_helpers():
    assert "queue()" not in worker._inject_preamble({}, wrangler=False)
    preamble = worker._inject_preamble({}, wrangler=True)
    for helper in ("def queue()", "def inspect(", "def claim_and_run(", "def send(", "_WRANGLER_USED"):
        assert helper in preamble


def test_wrangler_picks_and_runs_from_queue():
    subagents.run("pickme", "print(6*7)")
    wrangler = subagents.run(
        "wr1",
        "import json\n"
        "q = queue()\n"
        "assert q, 'queue not visible to wrangler'\n"
        "out = claim_and_run(q[0]['child'])\n"
        "print(json.dumps(out))",
        wrangler=True,
    )
    claimed = subagents.claim(wrangler["rlm_child_id"])
    result = worker._process_one(claimed, timeout=30, executor="deterministic", llm_opts={})

    assert result["ok"] is True
    payload = json.loads(result["text"].strip().splitlines()[-1])
    assert payload["ok"] is True
    assert "42" in payload.get("text", "")

    picked = [r for r in subagents.list_subagents() if r.get("name") == "pickme"][0]
    assert picked["status"] == "done"
    assert "42" in subagents.child_result(picked["rlm_child_id"]).get("result", "")


def test_wrangler_claim_cap_after_limit():
    for i in range(13):
        subagents.run(f"cap{i}", f"print({i})")
    wrangler = subagents.run(
        "wr2",
        "import json\n"
        "q = queue()\n"
        "results = [claim_and_run(t['child']) for t in q]\n"
        "caps = [r for r in results if not r.get('ok') and 'cap reached' in str(r.get('error', ''))]\n"
        "print(json.dumps({'done': sum(1 for r in results if r.get('ok')), 'caps': len(caps)}))",
        wrangler=True,
    )
    claimed = subagents.claim(wrangler["rlm_child_id"])
    result = worker._process_one(claimed, timeout=60, executor="deterministic", llm_opts={})
    assert result["ok"] is True
    payload = json.loads(result["text"].strip().splitlines()[-1])
    assert payload == {"done": 12, "caps": 1}
