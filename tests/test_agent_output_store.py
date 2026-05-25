"""Tests for agent-output-store."""
import threading
import pytest
from agent_output_store import OutputStore, Artifact, ArtifactNotFound


def test_store_and_get():
    store = OutputStore()
    a = store.store(run_id="run-1", kind="llm_response", data={"text": "hello"})
    got = store.get(a.artifact_id)
    assert got.data == {"text": "hello"}
    assert got.run_id == "run-1"


def test_get_missing():
    store = OutputStore()
    with pytest.raises(ArtifactNotFound):
        store.get("nonexistent")


def test_get_or_none():
    store = OutputStore()
    assert store.get_or_none("missing") is None


def test_size():
    store = OutputStore()
    store.store(kind="a", data=1)
    store.store(kind="b", data=2)
    assert store.size == 2


def test_list_all():
    store = OutputStore()
    store.store(kind="a", data=1)
    store.store(kind="b", data=2)
    assert len(store.list()) == 2


def test_list_by_run_id():
    store = OutputStore()
    store.store(run_id="run-1", kind="out", data=1)
    store.store(run_id="run-2", kind="out", data=2)
    results = store.list(run_id="run-1")
    assert len(results) == 1
    assert results[0].run_id == "run-1"


def test_list_by_kind():
    store = OutputStore()
    store.store(kind="llm_response", data="text")
    store.store(kind="tool_output", data="value")
    results = store.list(kind="llm_response")
    assert all(a.kind == "llm_response" for a in results)


def test_list_by_tag():
    store = OutputStore()
    store.store(kind="a", data=1, tags=["prod"])
    store.store(kind="b", data=2, tags=["dev"])
    results = store.list(tag="prod")
    assert len(results) == 1


def test_list_limit():
    store = OutputStore()
    for i in range(10):
        store.store(kind="out", data=i)
    results = store.list(limit=3)
    assert len(results) == 3


def test_search_by_kind():
    store = OutputStore()
    store.store(kind="llm_response", data="hello")
    store.store(kind="tool_output", data="world")
    results = store.search(kind="llm_response")
    assert len(results) == 1


def test_search_predicate():
    store = OutputStore()
    store.store(kind="out", data={"score": 0.9})
    store.store(kind="out", data={"score": 0.1})
    results = store.search(predicate=lambda a: a.data.get("score", 0) > 0.5)
    assert len(results) == 1
    assert results[0].data["score"] == 0.9


def test_delete():
    store = OutputStore()
    a = store.store(kind="out", data=1)
    assert store.delete(a.artifact_id) is True
    assert store.get_or_none(a.artifact_id) is None


def test_delete_missing():
    store = OutputStore()
    assert store.delete("nope") is False


def test_clear():
    store = OutputStore()
    store.store(kind="a", data=1)
    store.store(kind="b", data=2)
    count = store.clear()
    assert count == 2
    assert store.size == 0


def test_len():
    store = OutputStore()
    store.store(kind="a", data=1)
    assert len(store) == 1


def test_contains():
    store = OutputStore()
    a = store.store(kind="a", data=1)
    assert a.artifact_id in store
    assert "nope" not in store


def test_artifact_has_timestamp():
    store = OutputStore()
    a = store.store(kind="out", data=1)
    assert a.created_at > 0


def test_custom_artifact_id():
    store = OutputStore()
    a = store.store(kind="out", data=1, artifact_id="my-id")
    assert a.artifact_id == "my-id"
    got = store.get("my-id")
    assert got.data == 1


def test_persist_and_reload(tmp_path):
    path = str(tmp_path / "store.jsonl")
    s1 = OutputStore(path)
    a = s1.store(kind="out", data={"x": 1})
    s2 = OutputStore(path)
    got = s2.get(a.artifact_id)
    assert got.data == {"x": 1}


def test_thread_safe_concurrent_store():
    store = OutputStore()
    errors = []

    def worker():
        try:
            store.store(kind="out", data="value")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert store.size == 20
