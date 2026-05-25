"""
agent-output-store: JSONL artifact store for agent run outputs.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Artifact:
    artifact_id: str
    kind: str
    data: Any
    run_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "data": self.data,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=d["artifact_id"],
            kind=d["kind"],
            data=d["data"],
            run_id=d.get("run_id"),
            created_at=d.get("created_at", time.time()),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )


class ArtifactNotFound(KeyError):
    pass


class OutputStore:
    """
    Thread-safe JSONL artifact store for agent run outputs.

    Artifacts are appended to a JSONL file and kept in an in-memory index.
    Supports store, get, list, search, delete, and clear.

    Usage::

        store = OutputStore("/tmp/run.jsonl")
        store.store("run-1", "llm_response", {"content": "Hello!"}, tags=["prod"])
        artifact = store.get("run-1")
        results = store.search(kind="llm_response")
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._index: dict[str, Artifact] = {}
        self._lock = threading.Lock()
        self._counter = 0
        if path and os.path.exists(path):
            self._load(path)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    a = Artifact.from_dict(d)
                    self._index[a.artifact_id] = a
                except (json.JSONDecodeError, KeyError):
                    pass

    def _next_id(self) -> str:
        self._counter += 1
        return f"artifact-{self._counter:06d}"

    def store(
        self,
        run_id: Optional[str] = None,
        kind: str = "output",
        data: Any = None,
        tags: Optional[list[str]] = None,
        artifact_id: Optional[str] = None,
        **metadata: Any,
    ) -> Artifact:
        with self._lock:
            aid = artifact_id or self._next_id()
            artifact = Artifact(
                artifact_id=aid,
                kind=kind,
                data=data,
                run_id=run_id,
                tags=tags or [],
                metadata=metadata,
            )
            self._index[aid] = artifact
            if self._path:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(artifact.to_dict()) + "\n")
        return artifact

    def get(self, artifact_id: str) -> Artifact:
        a = self._index.get(artifact_id)
        if a is None:
            raise ArtifactNotFound(artifact_id)
        return a

    def get_or_none(self, artifact_id: str) -> Optional[Artifact]:
        return self._index.get(artifact_id)

    def list(
        self,
        run_id: Optional[str] = None,
        kind: Optional[str] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Artifact]:
        results = list(self._index.values())
        if run_id is not None:
            results = [a for a in results if a.run_id == run_id]
        if kind is not None:
            results = [a for a in results if a.kind == kind]
        if tag is not None:
            results = [a for a in results if tag in a.tags]
        results.sort(key=lambda a: a.created_at)
        if limit is not None:
            results = results[:limit]
        return results

    def search(
        self,
        kind: Optional[str] = None,
        run_id: Optional[str] = None,
        tag: Optional[str] = None,
        predicate: Optional[Callable[[Artifact], bool]] = None,
    ) -> list[Artifact]:
        results = self.list(run_id=run_id, kind=kind, tag=tag)
        if predicate is not None:
            results = [a for a in results if predicate(a)]
        return results

    def delete(self, artifact_id: str) -> bool:
        with self._lock:
            if artifact_id not in self._index:
                return False
            del self._index[artifact_id]
        return True

    def clear(self) -> int:
        with self._lock:
            count = len(self._index)
            self._index.clear()
            self._counter = 0
            if self._path and os.path.exists(self._path):
                open(self._path, "w").close()  # truncate
        return count

    @property
    def size(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, artifact_id: str) -> bool:
        return artifact_id in self._index


__all__ = ["OutputStore", "Artifact", "ArtifactNotFound"]
