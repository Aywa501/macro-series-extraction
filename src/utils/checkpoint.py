"""Lightweight JSON-backed checkpoint store for pipeline resumption."""

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class CheckpointStore:
    """Persist a set of completed keys to a JSON file on disk.

    Parameters
    ----------
    path:
        Path to the JSON checkpoint file. Parent directories must exist.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._done: set[str] = set()
        self.load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_done(self, key: str) -> bool:
        """Return True if *key* has previously been marked as done."""
        return key in self._done

    def mark_done(self, key: str) -> None:
        """Mark *key* as done and persist immediately."""
        self._done.add(key)
        self.save()

    def load(self) -> None:
        """Load checkpoint state from disk (no-op if file absent)."""
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._done = set(data.get("done", []))
                logger.debug("Loaded %d checkpoint keys from %s", len(self._done), self.path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load checkpoint %s: %s. Starting fresh.", self.path, exc)
                self._done = set()
        else:
            self._done = set()

    def save(self) -> None:
        """Persist current state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump({"done": sorted(self._done)}, fh, indent=2)

    def __len__(self) -> int:
        return len(self._done)

    def __repr__(self) -> str:
        return f"CheckpointStore(path={self.path!r}, n_done={len(self._done)})"
