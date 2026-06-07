"""Tests for agent-output-store.

These tests use only the Python standard library (``unittest``) so they run
with::

    python3 -m unittest discover -s tests
"""
import os
import tempfile
import threading
import unittest

from agent_output_store import (
    Artifact,
    ArtifactNotFound,
    OutputStore,
    __version__,
)


class StoreAndGetTests(unittest.TestCase):
    def test_store_and_get(self):
        store = OutputStore()
        a = store.store(run_id="run-1", kind="llm_response", data={"text": "hello"})
        got = store.get(a.artifact_id)
        self.assertEqual(got.data, {"text": "hello"})
        self.assertEqual(got.run_id, "run-1")

    def test_get_missing(self):
        store = OutputStore()
        with self.assertRaises(ArtifactNotFound):
            store.get("nonexistent")

    def test_artifact_not_found_is_keyerror(self):
        # ArtifactNotFound subclasses KeyError, so callers catching KeyError
        # keep working.
        self.assertTrue(issubclass(ArtifactNotFound, KeyError))

    def test_get_or_none(self):
        store = OutputStore()
        self.assertIsNone(store.get_or_none("missing"))

    def test_auto_ids_are_unique_and_sequential(self):
        store = OutputStore()
        a = store.store(kind="out", data=1)
        b = store.store(kind="out", data=2)
        self.assertNotEqual(a.artifact_id, b.artifact_id)
        self.assertEqual(a.artifact_id, "artifact-000001")
        self.assertEqual(b.artifact_id, "artifact-000002")

    def test_store_with_metadata(self):
        store = OutputStore()
        a = store.store(kind="out", data=1, source="agent", attempt=2)
        self.assertEqual(a.metadata, {"source": "agent", "attempt": 2})


class SizeAndContainerTests(unittest.TestCase):
    def test_size(self):
        store = OutputStore()
        store.store(kind="a", data=1)
        store.store(kind="b", data=2)
        self.assertEqual(store.size, 2)

    def test_len(self):
        store = OutputStore()
        store.store(kind="a", data=1)
        self.assertEqual(len(store), 1)

    def test_contains(self):
        store = OutputStore()
        a = store.store(kind="a", data=1)
        self.assertIn(a.artifact_id, store)
        self.assertNotIn("nope", store)


class ListTests(unittest.TestCase):
    def test_list_all(self):
        store = OutputStore()
        store.store(kind="a", data=1)
        store.store(kind="b", data=2)
        self.assertEqual(len(store.list()), 2)

    def test_list_by_run_id(self):
        store = OutputStore()
        store.store(run_id="run-1", kind="out", data=1)
        store.store(run_id="run-2", kind="out", data=2)
        results = store.list(run_id="run-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].run_id, "run-1")

    def test_list_by_kind(self):
        store = OutputStore()
        store.store(kind="llm_response", data="text")
        store.store(kind="tool_output", data="value")
        results = store.list(kind="llm_response")
        self.assertTrue(all(a.kind == "llm_response" for a in results))
        self.assertEqual(len(results), 1)

    def test_list_by_tag(self):
        store = OutputStore()
        store.store(kind="a", data=1, tags=["prod"])
        store.store(kind="b", data=2, tags=["dev"])
        results = store.list(tag="prod")
        self.assertEqual(len(results), 1)

    def test_list_limit(self):
        store = OutputStore()
        for i in range(10):
            store.store(kind="out", data=i)
        results = store.list(limit=3)
        self.assertEqual(len(results), 3)

    def test_list_sorted_by_created_at(self):
        store = OutputStore()
        a = store.store(kind="out", data=1)
        a.created_at = 100.0
        b = store.store(kind="out", data=2)
        b.created_at = 50.0
        results = store.list()
        self.assertEqual(results[0].data, 2)
        self.assertEqual(results[1].data, 1)


class SearchTests(unittest.TestCase):
    def test_search_by_kind(self):
        store = OutputStore()
        store.store(kind="llm_response", data="hello")
        store.store(kind="tool_output", data="world")
        results = store.search(kind="llm_response")
        self.assertEqual(len(results), 1)

    def test_search_predicate(self):
        store = OutputStore()
        store.store(kind="out", data={"score": 0.9})
        store.store(kind="out", data={"score": 0.1})
        results = store.search(predicate=lambda a: a.data.get("score", 0) > 0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data["score"], 0.9)

    def test_search_combines_filter_and_predicate(self):
        store = OutputStore()
        store.store(kind="out", data={"q": "python"}, tags=["final"])
        store.store(kind="out", data={"q": "rust"}, tags=["final"])
        store.store(kind="note", data={"q": "python"}, tags=["final"])
        results = store.search(
            kind="out",
            tag="final",
            predicate=lambda a: a.data.get("q") == "python",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data["q"], "python")


class DeleteAndClearTests(unittest.TestCase):
    def test_delete(self):
        store = OutputStore()
        a = store.store(kind="out", data=1)
        self.assertTrue(store.delete(a.artifact_id))
        self.assertIsNone(store.get_or_none(a.artifact_id))

    def test_delete_missing(self):
        store = OutputStore()
        self.assertFalse(store.delete("nope"))

    def test_clear(self):
        store = OutputStore()
        store.store(kind="a", data=1)
        store.store(kind="b", data=2)
        count = store.clear()
        self.assertEqual(count, 2)
        self.assertEqual(store.size, 0)


class ArtifactTests(unittest.TestCase):
    def test_artifact_has_timestamp(self):
        store = OutputStore()
        a = store.store(kind="out", data=1)
        self.assertGreater(a.created_at, 0)

    def test_custom_artifact_id(self):
        store = OutputStore()
        a = store.store(kind="out", data=1, artifact_id="my-id")
        self.assertEqual(a.artifact_id, "my-id")
        got = store.get("my-id")
        self.assertEqual(got.data, 1)

    def test_to_dict_round_trip(self):
        a = Artifact(
            artifact_id="x",
            kind="out",
            data={"k": "v"},
            run_id="run-1",
            tags=["t"],
            metadata={"m": 1},
        )
        restored = Artifact.from_dict(a.to_dict())
        self.assertEqual(restored.artifact_id, a.artifact_id)
        self.assertEqual(restored.kind, a.kind)
        self.assertEqual(restored.data, a.data)
        self.assertEqual(restored.run_id, a.run_id)
        self.assertEqual(restored.tags, a.tags)
        self.assertEqual(restored.metadata, a.metadata)

    def test_from_dict_defaults(self):
        restored = Artifact.from_dict(
            {"artifact_id": "x", "kind": "out", "data": 1}
        )
        self.assertIsNone(restored.run_id)
        self.assertEqual(restored.tags, [])
        self.assertEqual(restored.metadata, {})
        self.assertGreater(restored.created_at, 0)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "store.jsonl")

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, name))
        os.rmdir(self.tmpdir)

    def test_persist_and_reload(self):
        s1 = OutputStore(self.path)
        a = s1.store(kind="out", data={"x": 1})
        s2 = OutputStore(self.path)
        got = s2.get(a.artifact_id)
        self.assertEqual(got.data, {"x": 1})

    def test_reload_does_not_reuse_ids(self):
        # Regression test: after reloading a persisted store, the auto-id
        # counter must be advanced past the loaded artifacts so a new
        # store() call does not generate a colliding id and silently
        # overwrite an existing artifact.
        s1 = OutputStore(self.path)
        first = s1.store(kind="out", data="first")

        s2 = OutputStore(self.path)
        second = s2.store(kind="out", data="second")

        self.assertNotEqual(first.artifact_id, second.artifact_id)
        self.assertEqual(s2.size, 2)
        self.assertEqual(s2.get(first.artifact_id).data, "first")
        self.assertEqual(s2.get(second.artifact_id).data, "second")

    def test_custom_auto_style_id_advances_counter(self):
        # If a caller supplies an explicit id that looks like an auto id,
        # the next auto-generated id must not collide with it.
        store = OutputStore()
        store.store(kind="out", data=1, artifact_id="artifact-000005")
        nxt = store.store(kind="out", data=2)
        self.assertEqual(nxt.artifact_id, "artifact-000006")
        self.assertEqual(store.size, 2)

    def test_load_skips_corrupt_lines(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write('{"artifact_id": "ok", "kind": "out", "data": 1}\n')
            fh.write("not json at all\n")
            fh.write("\n")
            fh.write('{"missing": "required fields"}\n')
        store = OutputStore(self.path)
        self.assertEqual(store.size, 1)
        self.assertEqual(store.get("ok").data, 1)

    def test_clear_truncates_file(self):
        store = OutputStore(self.path)
        store.store(kind="out", data=1)
        store.clear()
        reloaded = OutputStore(self.path)
        self.assertEqual(reloaded.size, 0)


class ConcurrencyTests(unittest.TestCase):
    def test_thread_safe_concurrent_store(self):
        store = OutputStore()
        errors = []

        def worker():
            try:
                store.store(kind="out", data="value")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(store.size, 20)

    def test_concurrent_store_produces_unique_ids(self):
        store = OutputStore()

        def worker():
            store.store(kind="out", data="value")

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = [a.artifact_id for a in store.list()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(store.size, 50)


class PackageMetadataTests(unittest.TestCase):
    def test_version_is_exposed(self):
        self.assertIsInstance(__version__, str)
        self.assertTrue(__version__)


if __name__ == "__main__":
    unittest.main()
