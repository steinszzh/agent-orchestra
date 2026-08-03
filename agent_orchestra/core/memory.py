"""Shared memory (blackboard) for multi-agent state."""
from __future__ import annotations

import threading
from typing import Any, Optional

from .message import Message


class Memory:
    """Thread-safe shared blackboard.

    Agents can read/write arbitrary key-value state and the orchestrator
    keeps a full message history here so any agent can inspect prior turns.
    """

    def __init__(self) -> None:
        """Initialise an empty shared memory with a thread-safe lock."""
        self._store: dict[str, Any] = {}
        self._history: list[Message] = []
        self._lock = threading.RLock()

    # -- key/value store -------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def append(self, key: str, value: Any) -> None:
        """Append to a list-valued key (creating it if absent)."""
        with self._lock:
            self._store.setdefault(key, []).append(value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    # -- message history -------------------------------------------------
    def add_message(self, message: Message) -> None:
        with self._lock:
            self._history.append(message)

    @property
    def history(self) -> list[Message]:
        with self._lock:
            return list(self._history)

    def recent(self, n: int = 5) -> list[Message]:
        """Return the last *n* messages (most recent last)."""
        with self._lock:
            return list(self._history[-n:])

    # -- misc ------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the key/value store."""
        with self._lock:
            return dict(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._history.clear()

    def __contains__(self, key: str) -> bool:
        """Return True if *key* exists in the key/value store."""
        with self._lock:
            return key in self._store

    def __getitem__(self, key: str) -> Any:
        """Return the value for *key* (equivalent to :meth:`get`)."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set *key* to *value* (equivalent to :meth:`set`)."""