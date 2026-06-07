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
        art = store.store("run-1", "llm_response", {"content": "Hello!"}, tags=["prod"])
        artifact = store.get(art.artifact_id)
        results = store.search(kind="llm_response")
        finals = store.search(predicate=lambda a: "final" in a.tags)
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """Create a store, optionally backed by a JSONL file.

        If ``path`` is given and the file exists, existing artifacts are
        loaded into the in-memory index and the auto-id counter is advanced
        past any previously auto-generated ids so reloads never collide.

        :param path: Optional path to a JSONL file for persistence. When
            ``None``, the store is purely in-memory.
        """
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
                    self._advance_counter(a.artifact_id)
                except (json.JSONDecodeError, KeyError):
                    pass

    def _advance_counter(self, artifact_id: str) -> None:
        """Keep the auto-id counter ahead of any auto-generated id.

        Auto-generated ids have the form ``artifact-NNNNNN``. When ids of
        that form are loaded or supplied explicitly, the counter is bumped so
        the next auto-generated id cannot collide with an existing artifact.
        """
        if artifact_id.startswith("artifact-"):
            suffix = artifact_id[len("artifact-"):]
            if suffix.isdigit():
                self._counter = max(self._counter, int(suffix))

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
        """Store a new artifact and return it.

        The artifact is added to the in-memory index and, if the store is
        backed by a file, appended to the JSONL file. This method is
        thread-safe.

        :param run_id: Identifier of the agent run this artifact belongs to.
        :param kind: Category of the artifact (e.g. ``"llm_response"``).
        :param data: Arbitrary JSON-serializable payload.
        :param tags: Optional list of free-form tags.
        :param artifact_id: Optional explicit id. If omitted, an id of the
            form ``artifact-NNNNNN`` is generated automatically.
        :param metadata: Extra keyword arguments stored as metadata.
        :returns: The created :class:`Artifact`.
        """
        with self._lock:
            if artifact_id is None:
                aid = self._next_id()
            else:
                aid = artifact_id
                self._advance_counter(aid)
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
        """Return the artifact with the given id.

        :param artifact_id: The id of the artifact to retrieve.
        :raises ArtifactNotFound: If no artifact has that id.
        """
        a = self._index.get(artifact_id)
        if a is None:
            raise ArtifactNotFound(artifact_id)
        return a

    def get_or_none(self, artifact_id: str) -> Optional[Artifact]:
        """Return the artifact with the given id, or ``None`` if absent."""
        return self._index.get(artifact_id)

    def list(
        self,
        run_id: Optional[str] = None,
        kind: Optional[str] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Artifact]:
        """Return artifacts matching all of the given filters.

        Results are sorted by ``created_at`` (oldest first). All filters are
        optional; with no filters, every artifact is returned.

        :param run_id: Only include artifacts from this run.
        :param kind: Only include artifacts of this kind.
        :param tag: Only include artifacts carrying this tag.
        :param limit: Cap the number of results returned.
        """
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
        """Return artifacts matching the filters and an optional predicate.

        This is :meth:`list` plus an arbitrary ``predicate`` callable applied
        after the structured filters. Note that ``predicate`` is keyword-only
        in practice: the first positional argument is ``kind``.

        :param kind: Only include artifacts of this kind.
        :param run_id: Only include artifacts from this run.
        :param tag: Only include artifacts carrying this tag.
        :param predicate: Callable taking an :class:`Artifact` and returning
            ``True`` to keep it.
        """
        results = self.list(run_id=run_id, kind=kind, tag=tag)
        if predicate is not None:
            results = [a for a in results if predicate(a)]
        return results

    def delete(self, artifact_id: str) -> bool:
        """Remove an artifact from the in-memory index.

        :param artifact_id: The id of the artifact to remove.
        :returns: ``True`` if an artifact was removed, ``False`` if no
            artifact had that id.

        .. note::
            For file-backed stores the JSONL log is append-only, so the
            deleted record is not removed from the file. It is filtered out
            on the next reload only if it is still present in memory at the
            time the file was written. To physically compact the file, use
            :meth:`clear` or rewrite the store.
        """
        with self._lock:
            if artifact_id not in self._index:
                return False
            del self._index[artifact_id]
        return True

    def clear(self) -> int:
        """Remove all artifacts and truncate the backing file if any.

        :returns: The number of artifacts that were removed.
        """
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


__version__ = "0.1.0"

__all__ = ["OutputStore", "Artifact", "ArtifactNotFound", "__version__"]
