"""
agent-output-store: Persistent artifact store for agent outputs.

Stores named content artifacts with tags and metadata in a JSONL file.
Artifacts survive across runs. Supports tag filtering, substring search,
and thread-safe concurrent access.
"""

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


class ArtifactNotFoundError(Exception):
    """Raised when an artifact id is not found or has been deleted."""


@dataclass(frozen=True)
class Artifact:
    """An immutable artifact record."""

    id: str  # uuid4 hex
    name: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""  # ISO 8601 UTC
    metadata: dict = field(default_factory=dict)


class OutputStore:
    """
    Persistent artifact store backed by a JSONL file.

    Each line in the file is one of:
    - A full artifact JSON object (all fields)
    - {"_deleted": "<artifact_id>"}  — marks that id as deleted
    - {"_cleared": true}              — marks all prior artifacts as deleted

    On load the log is replayed in order; _cleared resets in-memory state,
    _deleted removes a specific id. This means appending is the only write
    operation, which keeps concurrent access simple and avoids corruption.
    """

    def __init__(self, persist_path: str) -> None:
        self._path = os.path.expanduser(persist_path)
        parent = os.path.dirname(self._path)
        os.makedirs(parent if parent else ".", exist_ok=True)
        self._lock = threading.RLock()
        # Ordered dict: id -> Artifact (non-deleted only)
        self._artifacts: dict[str, Artifact] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Replay the JSONL log into in-memory state."""
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # skip corrupt lines

                if "_cleared" in record:
                    self._artifacts.clear()
                elif "_deleted" in record:
                    self._artifacts.pop(record["_deleted"], None)
                else:
                    # Full artifact record
                    try:
                        artifact = Artifact(
                            id=record["id"],
                            name=record["name"],
                            content=record["content"],
                            tags=record.get("tags", []),
                            created_at=record.get("created_at", ""),
                            metadata=record.get("metadata", {}),
                        )
                        self._artifacts[artifact.id] = artifact
                    except (KeyError, TypeError):
                        continue  # skip malformed lines

    def _append(self, record: dict) -> None:
        """Append a JSON record to the JSONL file."""
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def store(
        self,
        name: str,
        content: str,
        tags: list[str] | None = None,
        **metadata,
    ) -> Artifact:
        """
        Create and persist a new artifact.

        Extra keyword arguments are stored in the artifact's metadata dict.
        Returns the created Artifact.
        """
        artifact = Artifact(
            id=uuid.uuid4().hex,
            name=name,
            content=content,
            tags=list(tags) if tags is not None else [],
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            metadata=dict(metadata),
        )
        record = {
            "id": artifact.id,
            "name": artifact.name,
            "content": artifact.content,
            "tags": artifact.tags,
            "created_at": artifact.created_at,
            "metadata": artifact.metadata,
        }
        with self._lock:
            self._artifacts[artifact.id] = artifact
            self._append(record)
        return artifact

    def get(self, artifact_id: str) -> Artifact:
        """
        Return the artifact with the given id.

        Raises ArtifactNotFoundError if it does not exist or was deleted.
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                raise ArtifactNotFoundError(artifact_id)
            return self._artifacts[artifact_id]

    def list(self, tag: str | None = None) -> list[Artifact]:
        """
        Return all non-deleted artifacts sorted newest-first.

        If tag is given, only return artifacts that include that tag.
        """
        with self._lock:
            artifacts = list(self._artifacts.values())

        if tag is not None:
            artifacts = [a for a in artifacts if tag in a.tags]

        # Sort by created_at descending; fall back to id for stable tie-breaking
        artifacts.sort(key=lambda a: (a.created_at, a.id), reverse=True)
        return artifacts

    def search(self, query: str) -> list[Artifact]:
        """
        Case-insensitive substring search across artifact name and content.

        An empty query returns all artifacts (same as list()).
        Results are sorted newest-first.
        """
        with self._lock:
            artifacts = list(self._artifacts.values())

        lower = query.lower()
        if lower:
            artifacts = [
                a for a in artifacts if lower in a.name.lower() or lower in a.content.lower()
            ]

        artifacts.sort(key=lambda a: (a.created_at, a.id), reverse=True)
        return artifacts

    def delete(self, artifact_id: str) -> bool:
        """
        Mark the artifact as deleted.

        Returns True if the artifact existed and was deleted, False if not found.
        """
        with self._lock:
            if artifact_id not in self._artifacts:
                return False
            del self._artifacts[artifact_id]
            self._append({"_deleted": artifact_id})
            return True

    def tags(self) -> list[str]:
        """Return sorted list of unique tags across all non-deleted artifacts."""
        with self._lock:
            all_tags: set[str] = set()
            for artifact in self._artifacts.values():
                all_tags.update(artifact.tags)
        return sorted(all_tags)

    def clear(self) -> int:
        """
        Mark all artifacts as deleted.

        Returns the count of artifacts that were deleted.
        Writes a single {"_cleared": true} sentinel to the JSONL log.
        """
        with self._lock:
            count = len(self._artifacts)
            self._artifacts.clear()
            self._append({"_cleared": True})
        return count

    def count(self) -> int:
        """Return the number of non-deleted artifacts."""
        with self._lock:
            return len(self._artifacts)
