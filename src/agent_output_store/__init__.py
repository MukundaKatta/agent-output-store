"""
agent-output-store: JSONL artifact store for agent run outputs.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_AUTO_ID_RE = re.compile(r"^artifact-(\d+)$")


@dataclass
class Artifact:
    artifact_id: str
    kind: str
    data: Any
    run_id: str | None = None
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
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
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
        art = store.store("run-1", "llm_response", {"content": "Hello!"}, tags=["prod"])
        artifact = store.get(art.artifact_id)
        results = store.search(kind="llm_response")
    """

    def __init__(self, path: str | None = None) -> None:
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
                    self._bump_counter(a.artifact_id)
                except (json.JSONDecodeError, KeyError):
                    pass

    def _bump_counter(self, artifact_id: str) -> None:
        """Advance the auto-id counter past any auto-generated id we've seen.

        Without this, reloading a store would reset the counter to 0 and cause
        newly generated ids to collide with (and overwrite) existing artifacts.
        """
        m = _AUTO_ID_RE.match(artifact_id)
        if m:
            self._counter = max(self._counter, int(m.group(1)))

    def _next_id(self) -> str:
        self._counter += 1
        return f"artifact-{self._counter:06d}"

    def _rewrite(self) -> None:
        """Persist the current in-memory index to disk, replacing the file.

        Used after deletions so removed artifacts do not reappear on reload.
        Caller must hold ``self._lock``.
        """
        if not self._path:
            return
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for artifact in self._index.values():
                fh.write(json.dumps(artifact.to_dict()) + "\n")
        os.replace(tmp, self._path)

    def store(
        self,
        run_id: str | None = None,
        kind: str = "output",
        data: Any = None,
        tags: list[str] | None = None,
        artifact_id: str | None = None,
        **metadata: Any,
    ) -> Artifact:
        with self._lock:
            if artifact_id is not None:
                aid = artifact_id
                self._bump_counter(aid)
            else:
                aid = self._next_id()
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

    def get_or_none(self, artifact_id: str) -> Artifact | None:
        return self._index.get(artifact_id)

    def list(
        self,
        run_id: str | None = None,
        kind: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
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
        kind: str | None = None,
        run_id: str | None = None,
        tag: str | None = None,
        predicate: Callable[[Artifact], bool] | None = None,
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
            if self._path:
                self._rewrite()
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
