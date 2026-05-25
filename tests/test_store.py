"""Tests for agent_output_store.OutputStore."""

import threading

import pytest

from agent_output_store import Artifact, ArtifactNotFoundError, OutputStore  # noqa: E402

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def store(tmp_path):
    return OutputStore(str(tmp_path / "store.jsonl"))


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "store.jsonl")


# ------------------------------------------------------------------ #
# store()
# ------------------------------------------------------------------ #


def test_store_returns_artifact(store):
    art = store.store("report", "hello world")
    assert isinstance(art, Artifact)


def test_store_correct_name_and_content(store):
    art = store.store("my-report", "some content")
    assert art.name == "my-report"
    assert art.content == "some content"


def test_store_default_tags_empty(store):
    art = store.store("x", "y")
    assert art.tags == []


def test_store_tags_stored(store):
    art = store.store("x", "y", tags=["alpha", "beta"])
    assert art.tags == ["alpha", "beta"]


def test_store_extra_kwargs_in_metadata(store):
    art = store.store("x", "y", source="pipeline", version=2)
    assert art.metadata["source"] == "pipeline"
    assert art.metadata["version"] == 2


def test_store_has_nonempty_id(store):
    art = store.store("x", "y")
    assert art.id and len(art.id) == 32  # uuid4 hex


def test_store_unique_ids(store):
    ids = {store.store("x", "y").id for _ in range(20)}
    assert len(ids) == 20


def test_store_created_at_nonempty(store):
    art = store.store("x", "y")
    assert art.created_at  # non-empty ISO string


def test_store_created_at_utc(store):
    art = store.store("x", "y")
    assert "+00:00" in art.created_at or art.created_at.endswith("Z")


# ------------------------------------------------------------------ #
# get()
# ------------------------------------------------------------------ #


def test_get_returns_correct_artifact(store):
    art = store.store("report", "body")
    fetched = store.get(art.id)
    assert fetched.id == art.id
    assert fetched.name == art.name
    assert fetched.content == art.content


def test_get_unknown_id_raises(store):
    with pytest.raises(ArtifactNotFoundError):
        store.get("doesnotexist")


# ------------------------------------------------------------------ #
# list()
# ------------------------------------------------------------------ #


def test_list_all_artifacts(store):
    store.store("a", "1")
    store.store("b", "2")
    store.store("c", "3")
    result = store.list()
    assert len(result) == 3


def test_list_newest_first(store):
    a = store.store("a", "1")
    b = store.store("b", "2")
    c = store.store("c", "3")
    ids = [art.id for art in store.list()]
    # c was stored last so should appear first
    assert ids.index(c.id) < ids.index(b.id) < ids.index(a.id)


def test_list_with_tag_filters(store):
    store.store("a", "1", tags=["x"])
    store.store("b", "2", tags=["y"])
    store.store("c", "3", tags=["x", "y"])
    result = store.list(tag="x")
    assert len(result) == 2
    assert all("x" in art.tags for art in result)


def test_list_tag_no_match_returns_empty(store):
    store.store("a", "1", tags=["foo"])
    result = store.list(tag="bar")
    assert result == []


# ------------------------------------------------------------------ #
# search()
# ------------------------------------------------------------------ #


def test_search_by_name_substring(store):
    store.store("monthly-report", "content")
    store.store("daily-summary", "content")
    result = store.search("monthly")
    assert len(result) == 1
    assert result[0].name == "monthly-report"


def test_search_by_content_substring(store):
    store.store("artifact", "the final answer is 42")
    store.store("other", "nothing here")
    result = store.search("final answer")
    assert len(result) == 1
    assert "final answer" in result[0].content


def test_search_case_insensitive_name(store):
    store.store("Report-Alpha", "content")
    result = store.search("report-alpha")
    assert len(result) == 1


def test_search_case_insensitive_content(store):
    store.store("x", "Hello World")
    result = store.search("HELLO")
    assert len(result) == 1


def test_search_empty_query_returns_all(store):
    store.store("a", "1")
    store.store("b", "2")
    result = store.search("")
    assert len(result) == 2


def test_search_no_match_returns_empty(store):
    store.store("a", "content")
    result = store.search("zzz-not-here")
    assert result == []


def test_search_sorted_newest_first(store):
    a = store.store("common-name", "1")
    b = store.store("common-name", "2")
    result = store.search("common-name")
    assert result[0].id == b.id
    assert result[1].id == a.id


# ------------------------------------------------------------------ #
# delete()
# ------------------------------------------------------------------ #


def test_delete_returns_true_if_existed(store):
    art = store.store("x", "y")
    assert store.delete(art.id) is True


def test_delete_returns_false_if_not_found(store):
    assert store.delete("no-such-id") is False


def test_deleted_artifact_not_in_list(store):
    art = store.store("x", "y")
    store.delete(art.id)
    assert art.id not in [a.id for a in store.list()]


def test_get_after_delete_raises(store):
    art = store.store("x", "y")
    store.delete(art.id)
    with pytest.raises(ArtifactNotFoundError):
        store.get(art.id)


# ------------------------------------------------------------------ #
# tags()
# ------------------------------------------------------------------ #


def test_tags_sorted_unique(store):
    store.store("a", "1", tags=["z", "a", "m"])
    store.store("b", "2", tags=["a", "b"])
    result = store.tags()
    assert result == sorted(set(result))
    assert set(result) == {"z", "a", "m", "b"}


def test_tags_excludes_deleted_artifacts(store):
    art = store.store("a", "1", tags=["exclusive"])
    store.store("b", "2", tags=["common"])
    store.delete(art.id)
    assert "exclusive" not in store.tags()
    assert "common" in store.tags()


def test_tags_empty_when_no_artifacts(store):
    assert store.tags() == []


# ------------------------------------------------------------------ #
# clear()
# ------------------------------------------------------------------ #


def test_clear_returns_count(store):
    store.store("a", "1")
    store.store("b", "2")
    store.store("c", "3")
    assert store.clear() == 3


def test_clear_all_deleted(store):
    store.store("a", "1")
    store.store("b", "2")
    store.clear()
    assert store.list() == []


def test_clear_count_zero_after(store):
    store.store("a", "1")
    store.clear()
    assert store.count() == 0


# ------------------------------------------------------------------ #
# count()
# ------------------------------------------------------------------ #


def test_count_correct(store):
    assert store.count() == 0
    store.store("a", "1")
    assert store.count() == 1
    store.store("b", "2")
    assert store.count() == 2


def test_count_decreases_after_delete(store):
    art = store.store("a", "1")
    store.delete(art.id)
    assert store.count() == 0


# ------------------------------------------------------------------ #
# Persistence (reload)
# ------------------------------------------------------------------ #


def test_persist_store_survives_reload(store_path):
    s1 = OutputStore(store_path)
    art = s1.store("persisted", "value", tags=["p"])
    s2 = OutputStore(store_path)
    fetched = s2.get(art.id)
    assert fetched.name == "persisted"
    assert fetched.content == "value"
    assert fetched.tags == ["p"]


def test_persist_delete_survives_reload(store_path):
    s1 = OutputStore(store_path)
    art = s1.store("to-delete", "content")
    s1.delete(art.id)
    s2 = OutputStore(store_path)
    with pytest.raises(ArtifactNotFoundError):
        s2.get(art.id)


def test_persist_clear_survives_reload(store_path):
    s1 = OutputStore(store_path)
    s1.store("a", "1")
    s1.store("b", "2")
    s1.clear()
    s2 = OutputStore(store_path)
    assert s2.count() == 0


def test_persist_post_clear_store_survives_reload(store_path):
    """Artifacts added after a clear should survive reload."""
    s1 = OutputStore(store_path)
    s1.store("before", "x")
    s1.clear()
    art = s1.store("after", "y")
    s2 = OutputStore(store_path)
    assert s2.count() == 1
    fetched = s2.get(art.id)
    assert fetched.name == "after"


# ------------------------------------------------------------------ #
# Path handling
# ------------------------------------------------------------------ #


def test_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = OutputStore("~/tilde-store/store.jsonl")
    art = s.store("x", "y")
    assert s.get(art.id).name == "x"


def test_parent_dirs_auto_created(tmp_path):
    deep = str(tmp_path / "a" / "b" / "c" / "store.jsonl")
    s = OutputStore(deep)
    art = s.store("x", "y")
    assert s.get(art.id).id == art.id


# ------------------------------------------------------------------ #
# Thread safety
# ------------------------------------------------------------------ #


def test_thread_safety_concurrent_stores(store):
    results = []
    errors = []

    def worker():
        try:
            for _ in range(10):
                art = store.store("thread-art", "data")
                results.append(art.id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    assert len(results) == 100  # 10 threads * 10 stores
    assert len(set(results)) == 100  # all ids unique
    assert store.count() == 100
