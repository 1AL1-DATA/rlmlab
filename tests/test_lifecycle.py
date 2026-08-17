from rlmlab import harness, kernel, subagents


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

    # snapshots are taken at version % 10 == 1; drive to one and record its state
    while h.get_state()["version"] % harness.SNAPSHOT_INTERVAL != 1:
        h.refine("prompt_notes", "fill")
    snap_version = h.get_state()["version"]
    expected = h.get_state()
    ok = h.rollback(snap_version)
    assert ok["ok"] is True
    state = h.get_state()
    assert state["version"] == snap_version
    assert state["prompt_notes"] == expected["prompt_notes"]
    assert state["memories"] == expected["memories"]


def test_kernel_snapshot_restore():
    kernel.start("snapmem")
    try:
        r = kernel.exec_code("snapmem", "import pandas as pd\ndf = pd.DataFrame({'a':[1,2]})\nnotes = {'k': 42}")
        assert r["ok"] is True
        snap = kernel.snapshot("snapmem")
        assert snap["ok"] is True, snap
        assert snap["saved"] >= 2
    finally:
        kernel.stop("snapmem", snapshot_state=False)

    # simulate death: kernel stopped with snapshot, then restart revives
    assert kernel._read_index("snapmem") is None
    kernel.start("snapmem")
    try:
        r = kernel.exec_code("snapmem", "print(df.shape, notes['k'])")
        assert "2, 1" in r["text"]
        assert "42" in r["text"]
    finally:
        kernel.stop("snapmem", snapshot_state=False)
    # snapshot_on_stop is covered by reap; explicit snapshot above is the revive source
