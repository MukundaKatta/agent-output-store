# agent-output-store

[![CI](https://github.com/MukundaKatta/agent-output-store/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/agent-output-store/actions/workflows/ci.yml)

A small, dependency-free, thread-safe JSONL artifact store for agent run
outputs. Use it to capture the intermediate and final artifacts an agent
produces during a run (LLM responses, tool outputs, summaries) and query them
later by run, kind, tag, or arbitrary predicate.

- **Zero dependencies** — pure Python standard library.
- **Thread-safe** — concurrent writes from multiple worker threads are safe.
- **Optionally persistent** — back the store with a JSONL file, or keep it
  purely in memory.
- **Typed** — ships with inline type hints and a `py.typed` marker.

## Install

```
pip install agent-output-store
```

Requires Python 3.9+.

## Usage

```python
from agent_output_store import OutputStore

# In-memory store, or pass a path to persist to a JSONL file.
store = OutputStore("/tmp/agent-outputs.jsonl")

# Store artifacts. store() returns the created Artifact.
art = store.store("run-123", "search_result", {"query": "python", "results": []})
final = store.store("run-123", "summary", "Final answer here", tags=["final"])

# Retrieve a single artifact by its id.
same = store.get(art.artifact_id)            # raises ArtifactNotFound if absent
maybe = store.get_or_none("does-not-exist")  # returns None if absent

# List artifacts (sorted oldest-first), with optional filters.
arts = store.list("run-123")
summaries = store.list("run-123", kind="summary")
finals = store.list("run-123", tag="final")
recent = store.list(limit=10)

# Search with a predicate (note: predicate is a keyword argument).
matches = store.search(predicate=lambda a: a.data.get("query") == "python")
# Filters and predicate can be combined:
matches = store.search(kind="search_result",
                       predicate=lambda a: a.data.get("query") == "python")

# Containers and sizing.
len(store)                  # number of artifacts
art.artifact_id in store    # membership test
store.size                  # same as len(store)

# Delete and clear.
store.delete(art.artifact_id)   # True if removed, False if it wasn't there
store.clear()                   # remove everything; returns the count removed
```

## Persistence

When you pass a file path, every `store()` call appends one JSON object per
line (JSONL). On startup the store loads any existing artifacts from that file
back into memory.

Auto-generated ids have the form `artifact-NNNNNN`. When a store is reloaded
from a file, the internal counter is advanced past the highest existing id so
that new artifacts never collide with — or silently overwrite — ones already on
disk.

The JSONL log is append-only, so `delete()` removes an artifact from the
in-memory index but does not rewrite the file. Use `clear()` to truncate the
backing file.

## API

### `OutputStore(path: str | None = None)`

Create a store. If `path` is given and the file exists, its artifacts are
loaded into memory.

| Method | Description |
| --- | --- |
| `store(run_id=None, kind="output", data=None, tags=None, artifact_id=None, **metadata) -> Artifact` | Store and return a new artifact. Thread-safe; appends to the file if persistent. |
| `get(artifact_id) -> Artifact` | Return an artifact by id, or raise `ArtifactNotFound`. |
| `get_or_none(artifact_id) -> Artifact | None` | Return an artifact by id, or `None`. |
| `list(run_id=None, kind=None, tag=None, limit=None) -> list[Artifact]` | Return matching artifacts, sorted by `created_at`. |
| `search(kind=None, run_id=None, tag=None, predicate=None) -> list[Artifact]` | Like `list`, plus an arbitrary predicate callable. |
| `delete(artifact_id) -> bool` | Remove from the in-memory index. `True` if removed. |
| `clear() -> int` | Remove all artifacts (and truncate the file). Returns the count removed. |
| `size` / `len(store)` | Number of artifacts. |
| `artifact_id in store` | Membership test. |

### `Artifact`

A dataclass with fields `artifact_id`, `kind`, `data`, `run_id`,
`created_at`, `tags`, and `metadata`, plus `to_dict()` / `from_dict()` for
serialization.

### `ArtifactNotFound`

Raised by `get()` when no artifact has the requested id. Subclasses
`KeyError`.

## Development

Run the test suite with the standard library — no extra dependencies needed:

```
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
