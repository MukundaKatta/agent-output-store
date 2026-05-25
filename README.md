# agent-output-store

Thread-safe JSONL artifact store for agent run outputs.

## Install

```
pip install agent-output-store
```

## Usage

```python
from agent_output_store import OutputStore

store = OutputStore("/tmp/agent-outputs.jsonl")

# Store artifacts
art = store.store("run-123", "search_result", {"query": "python", "results": [...]})
art = store.store("run-123", "summary", "Final answer here", tags=["final"])

# List artifacts for a run
arts = store.list("run-123")
arts = store.list("run-123", kind="summary")
arts = store.list("run-123", tag="final")

# Search with predicate
arts = store.search(lambda a: a.data.get("query") == "python")

# Delete
store.delete(art.artifact_id)
```
