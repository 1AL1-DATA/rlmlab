from rlmlab import harness, subagents


def test_run_claim_complete_lifecycle():
    created = subagents.run("t", "print(1)")
    cid = created["rlm_child_id"]
    assert created["status"] == "admitted"
    assert subagents.list_subagents()[0]["status"] == "admitted"

    claimed = subagents.claim(cid)
    assert claimed["status"] == "running"
    assert claimed.get("started_ts")

    assert subagents.claim(cid) is None

    subagents.complete(cid, "42")
    rec = subagents._find(cid)
    assert rec["status"] == "done"
    assert rec.get("completed_ts")

    msgs = subagents.mailbox(cid)["messages"]
    assert msgs[-1]["from"] == "worker"
    assert msgs[-1]["text"] == "42"


def test_complete_error_sets_failed():
    created = subagents.run("t2", "boom")
    cid = created["rlm_child_id"]
    subagents.claim(cid)
    subagents.complete(cid, "", error="kaboom")
    assert subagents._find(cid)["status"] == "failed"
    assert subagents._find(cid)["error"] == "kaboom"


def test_depth_and_parent_tracking():
    parent = subagents.run("p", "x", parent=None, depth=0)
    child = subagents.run("c", "y", parent=parent["rlm_child_id"], depth=1)
    assert child["depth"] == 1
    assert child["parent"] == parent["rlm_child_id"]


def test_child_result():
    cid = subagents.run("cr", "z")["rlm_child_id"]
    r = subagents.child_result(cid)
    assert r["status"] == "admitted"
    assert r["result"] is None
    subagents.claim(cid)
    subagents.complete(cid, "done")
    r = subagents.child_result(cid)
    assert r["status"] == "done"
    assert r["result"] == "done"


def test_harness_refine_and_rollback():
    h = harness
    v0 = h.get_state()["version"]
    h.refine("prompt_notes", "note-a")
    h.refine("memories", "mem-b")
    state = h.get_state()
    assert state["version"] == v0 + 2
    assert state["prompt_notes"][-1]["text"] == "note-a"

    ok = h.rollback(v0)
    assert ok["ok"] is True
    state = h.get_state()
    assert state["version"] == v0
    assert state["prompt_notes"] == []
