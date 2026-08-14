import json

from rlmlab import apis, subagents, worker


def test_run_stores_api_field():
    rec = subagents.run("t", "print(1)", api="ollama")
    assert rec["api"] == "ollama"
    with open(rec["session_dir"] + "/admission.json") as f:
        stored = json.load(f)
    assert stored["api"] == "ollama"


def test_run_defaults_api_to_deterministic():
    rec = subagents.run("t2", "print(1)")
    assert rec["api"] == "deterministic"


def test_set_api_reassigns_admitted_only():
    rec = subagents.run("t3", "print(1)", api="deterministic")
    ok = subagents.set_api(rec["rlm_child_id"], "ollama")
    assert ok["ok"] is True
    assert ok["api"] == "ollama"
    claimed = subagents.claim(rec["rlm_child_id"])
    assert claimed is not None
    bad = subagents.set_api(rec["rlm_child_id"], "llama-8080")
    assert bad["ok"] is False
    assert "admitted" in bad["error"]


def test_apis_register_deregister():
    ok = apis.register("test-llm", "ollama", {"base_url": "http://127.0.0.1:9999", "model": "m"})
    assert ok["ok"] is True
    names = {a["name"] for a in apis.list_apis()["apis"]}
    assert "test-llm" in names
    assert apis.resolve("test-llm") == {
        "executor": "llm",
        "base_url": "http://127.0.0.1:9999",
        "model": "m",
    }
    assert apis.deregister("test-llm")["ok"] is True
    assert apis.deregister("deterministic")["ok"] is False


def test_apis_register_rejects_bad_type():
    assert apis.register("x", "bogus")["ok"] is False


def test_work_once_filters_by_api():
    a = subagents.run("fa", "print('a')", api="deterministic")["rlm_child_id"]
    b = subagents.run("fb", "print('b')", api="ollama")["rlm_child_id"]
    res = worker.work_once(api="deterministic")
    assert res["ok"] is True
    assert [r["child"] for r in res["processed"]] == [a]
    assert subagents.child_result(a)["status"] == "done"
    assert subagents.child_result(b)["status"] == "admitted"
