"""Agent base class and capability model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import LLMBackend, MockBackend
from .memory import Memory
from .message import Message, MessageRole, Task


@dataclass
class AgentCapability:
    """Declares something an agent can do (used for routing decisions)."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        """Return True if *text* contains any of this capability's keywords.

        The match is case-insensitive.  A keyword is considered a match
        when it appears as a substring of *text*.

        Args:
            text: The text to search for keyword matches.

        Returns:
            True if at least one keyword is found in *text*.
        """
        t = text.lower()
        return any(k.lower() in t for k in self.keywords)


class Agent:
    """Base agent.

    Subclass and override :meth:`process` / :meth:`build_prompt` for custom
    behaviour, or just set a ``system_prompt`` and rely on the default
    LLM-driven ``process``.
    """

    def __init__(
        self,
        name: str,
        role: str,
        *,
        description: str = "",
        system_prompt: Optional[str] = None,
        llm: Optional[LLMBackend] = None,
        capabilities: Optional[list[AgentCapability]] = None,
        shared_memory: Optional[Memory] = None,
    ) -> None:
        """Initialise an Agent.

        Args:
            name: Human-readable name for this agent.
            role: The agent's role (e.g. "research analyst", "content writer").
            description: A short description of what this agent does.
                Used in the default system prompt and for routing.
            system_prompt: The system prompt sent to the LLM.  If omitted,
                a default is generated from ``name``, ``role``, and
                ``description``.
            llm: The LLM backend to use.  Defaults to :class:`MockBackend`
                when none is provided.
            capabilities: A list of :class:`AgentCapability` instances that
                describe what this agent can do.  Used for routing decisions.
            shared_memory: A shared :class:`Memory` instance for inter-agent
                communication.  A new :class:`Memory` is created when
                ``None``.
        """
        self.id = name.lower().replace(" ", "_")
        self.name = name
        self.role = role
        self.description = description
        self.llm = llm or MockBackend()
        self.capabilities = capabilities or []
        self.memory = shared_memory or Memory()
        self.system_prompt = system_prompt or self._default_system_prompt()

    # ------------------------------------------------------------------
    def _default_system_prompt(self) -> str:
        caps = ", ".join(c.name for c in self.capabilities) or "general tasks"
        return (
            f"You are {self.name}, a {self.role}. "
            f"{self.description} You can handle: {caps}."
        )

    # ------------------------------------------------------------------
    def build_prompt(self, message: Message) -> str:
        """Construct the prompt sent to the LLM for *message*.

        Override to inject memory, retrieved context, etc.
        """
        parts: list[str] = []
        # Include recent shared context if available.
        if len(self.memory.history) > 1:
            parts.append("Recent context:")
            for m in self.memory.recent(3):
                if m.sender != self.id:
                    parts.append(f"  [{m.sender}] {m.content[:200]}")
            parts.append("")
        parts.append(f"Task: {message.content}")
        return "\n".join(parts)

    def process(self, message: Message) -> Message:
        """Handle an incoming message and return a response message."""
        prompt = self.build_prompt(message)
        content = self.llm.generate(prompt, system_prompt=self.system_prompt)
        return Message(
            content=content,
            role=MessageRole.ASSISTANT,
            sender=self.id,
            recipient=message.sender,
            metadata={**message.metadata, "agent": self.id},
        )

    # ------------------------------------------------------------------
    def can_handle(self, task: Task) -> bool:
        """Return True if this agent is capable of handling *task*."""
        if not self.capabilities:
            return True
        return any(c.matches(task.description) for c in self.capabilities) or True

    def handle_task(self, task: Task) -> Message:
        """Convenience: process a :class:`Task` directly."""
        msg = task.to_message(sender="orchestrator")
        msg.recipient = self.id
        return self.process(msg)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<Agent {self.name!r} role={self.role!r} caps={len(self.capabilities)}>"