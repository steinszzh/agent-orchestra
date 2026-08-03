"""Message and Task data structures for inter-agent communication."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """Roles a message can take in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message exchanged between agents (or user <-> agent).

    Attributes:
        content: The textual payload of the message.
        role: Conversational role (system/user/assistant/tool).
        sender: Identifier of the sending agent (or "user"/"orchestrator").
        recipient: Identifier of the target agent, or None for broadcast.
        metadata: Free-form dict for extra context (task ids, tags, etc.).
        timestamp: When the message was created.
        id: Unique message identifier.
    """

    content: str
    role: MessageRole = MessageRole.ASSISTANT
    sender: str = "system"
    recipient: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "sender": self.sender,
            "recipient": self.recipient,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        to = self.recipient or "all"
        return f"Message({self.sender}->{to}, role={self.role.value}, len={len(self.content)})"


@dataclass
class Task:
    """A unit of work to be dispatched to one or more agents.

    Attributes:
        description: Human-readable description of what needs doing.
        assigned_to: Agent id that should handle this task, or None.
        priority: Lower numbers run first when scheduling.
        context: Extra context dict forwarded to the handling agent.
        id: Unique task identifier.
    """

    description: str
    assigned_to: Optional[str] = None
    priority: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_message(self, sender: str = "orchestrator") -> Message:
        """Convert this task into a user-role Message for an agent."""
        return Message(
            content=self.description,
            role=MessageRole.USER,
            sender=sender,
            recipient=self.assigned_to,
            metadata={"task_id": self.id, **self.context},
        )