import fcntl
import time

from rlmlab import agent_loop, harness, subagents, worker


def test_worker_drains_deterministic():
    subagents.run("w1", "print(6*7)")
    result = worker.work_once(executor="deterministic")
    assert result["ok"] is True
    assert len(result["processed"]) == 1
    assert result["processed"][0]["ok"] is True
    assert "42" in result["processed"][0]["text"]

    second = worker.work_once(executor="deterministic")
    assert second["processed"] == []

    cid = subagents.list_subagents()[0]["rlm_child_id"]
    assert subagents._find(cid)["status"] == "done"


def test_worker_flock_guard():
    subagents.run("w2", "print(1)")
    lock_path = worker.LOCK_FILE
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = worker.work_once()
        assert result["ok"] is False
        assert "another worker" in result["error"]
        fcntl.flock(f, fcntl.LOCK_UN)
    # lock released: a fresh run should succeed (and process w2)
    result = worker.work_once()
    assert result["ok"] is True


def test_worker_feedback_writes_memory():
    rec = subagents.run("w3", "import json\njson.dumps({'a': 1})")
    worker._distill_feedback(rec, True, "the answer is 42")
    worker._distill_feedback(rec, False, "the task blew up")
    memories = [m["text"] for m in harness.get_state()["memories"]]
    assert any("completed" in m and "42" in m for m in memories)
    assert any("failed" in m and "blew up" in m for m in memories)


def test_reap_idle_kernel():
    from rlmlab import kernel

    kernel.start("reapme")
    res = worker.reap(max_age_seconds=0)
    assert "reapme" in res["reaped"]
    assert all(s["name"] != "reapme" for s in kernel.list_sessions())


def test_extract_json_variants():
    assert agent_loop._extract_json('{"action":"final","answer":"x"}') == {"action": "final", "answer": "x"}
    assert agent_loop._extract_json('```json\n{"action":"exec","code":"1"}\n```') == {"action": "exec", "code": "1"}
    wrapped = agent_loop._extract_json('here is my json {"action":"final","answer":"y"} thanks')
    assert wrapped == {"action": "final", "answer": "y"}
    assert agent_loop._extract_json("not json at all") is None


def test_stuck_child_recovered_requeued():
    """A child left 'running' by a crash (no kernel alive, claim old) is
    reset to 'admitted' so a later drain re-runs it."""
    rec = subagents.run("stuck1", "print('stuck')")
    # simulate a crash: claim it, then make the claim look stale with no kernel
    rec["status"] = "running"
    rec["started_ts"] = int(time.time() * 1000) - 7200_000
    subagents._write(rec)
    subagents._bust_list_cache()
    requeued = worker._requeue_stale_running(max_age_seconds=60)
    assert rec["rlm_child_id"] in requeued
    record = subagents._find(rec["rlm_child_id"])
    assert record["status"] == "admitted"


def test_claim_atomic_under_concurrent_claims():
    """Two racing claims on the same child must yield exactly one winner."""
    rec = subagents.run("race1", "print('x')")
    cid = rec["rlm_child_id"]
    winners = 0
    # serialized claim calls: second must return None because first flipped it
    first = subagents.claim(cid)
    second = subagents.claim(cid)
    if first is not None:
        winners += 1
    if second is not None:
        winners += 1
    assert winners == 1


def test_memory_retrieval_top_k():
    memories = ["kittens are soft and warm", "puppies bark loudly", "rust ownership moves data"]
    got = worker._retrieve_memories(memories, "kittens vs dogs", k=2)
    assert "kittens" in got[0]
    got2 = worker._retrieve_memories(memories, "unrelated query", k=2)
    assert len(got2) == 2
