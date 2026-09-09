"""Agent base class, capability model, and tool calling."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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


@dataclass
class Tool:
    """A callable capability an agent can invoke during processing.

    The LLM requests a call by emitting a single-line ``TOOL_CALL`` marker
    followed by the tool name and JSON arguments::

        TOOL_CALL: get_weather({"city": "Shanghai"})

    The framework parses the marker, executes :attr:`function` with the
    decoded arguments, and feeds the string result back into the next LLM
    round so the agent can answer using real tool output.
    """

    name: str
    description: str
    function: Callable[..., str]
    parameters: dict[str, Any] = field(default_factory=dict)

    def run(self, **kwargs: Any) -> str:
        """Invoke the underlying function with *kwargs*."""
        return self.function(**kwargs)


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
        tools: Optional[list[Tool]] = None,
        max_tool_rounds: int = 3,
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
            tools: Optional list of :class:`Tool` instances the agent can
                invoke.  Tools are advertised in the default system prompt.
            max_tool_rounds: Maximum number of tool-execution rounds per
                :meth:`process` call (guards against infinite tool loops).
        """
        self.id = name.lower().replace(" ", "_")
        self.name = name
        self.role = role
        self.description = description
        self.llm = llm or MockBackend()
        self.capabilities = capabilities or []
        self.tools = tools or []
        self.max_tool_rounds = max(1, max_tool_rounds)
        self.memory = shared_memory or Memory()
        self.system_prompt = system_prompt or self._default_system_prompt()

    # ------------------------------------------------------------------
    def _default_system_prompt(self) -> str:
        caps = ", ".join(c.name for c in self.capabilities) or "general tasks"
        parts = [
            f"You are {self.name}, a {self.role}. ",
            f"{self.description} You can handle: {caps}.",
        ]
        if self.tools:
            parts.append("\nAvailable tools:")
            for t in self.tools:
                parts.append(f"- {t.name}: {t.description}")
            parts.append(
                'To call a tool, reply with exactly one line: '
                'TOOL_CALL: <tool_name>({"arg": "value"})'
            )
        return "\n".join(parts)

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
        """Handle an incoming message and return a response message.

        If the agent has tools, the LLM output is scanned for ``TOOL_CALL``
        markers; matching tools are executed and their results are fed back
        into subsequent LLM rounds until no more calls are requested (or
        ``max_tool_rounds`` is reached).
        """
        prompt = self.build_prompt(message)
        tool_results: list[str] = []
        content = ""

        for _ in range(self.max_tool_rounds + 1):
            full_prompt = self._with_tool_results(prompt, tool_results)
            content = self.llm.generate(full_prompt, system_prompt=self.system_prompt)
            calls = self._parse_tool_calls(content)
            if not calls:
                break
            tool_results.extend(self._execute_tools(calls))

        return Message(
            content=content,
            role=MessageRole.ASSISTANT,
            sender=self.id,
            recipient=message.sender,
            metadata={**message.metadata, "agent": self.id},
        )

    # ------------------------------------------------------------------
    # Tool calling helpers
    # ------------------------------------------------------------------
    _TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\w+)\s*\((\{.*?\})\)", re.DOTALL)

    @classmethod
    def _parse_tool_calls(cls, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Extract ``(name, args)`` pairs from *text*.

        Only well-formed markers with valid JSON object arguments are kept.
        """
        calls: list[tuple[str, dict[str, Any]]] = []
        for m in cls._TOOL_CALL_RE.finditer(text):
            name = m.group(1)
            raw = m.group(2)
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(args, dict):
                calls.append((name, args))
        return calls

    def _execute_tools(self, calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Run each requested tool and return formatted result lines."""
        by_name = {t.name: t for t in self.tools}
        results: list[str] = []
        for name, args in calls:
            tool = by_name.get(name)
            if tool is None:
                results.append(f"TOOL_CALL: {name} -> ERROR: unknown tool")
                continue
            try:
                out = tool.run(**args)
                results.append(f"TOOL_CALL: {name} -> {out}")
            except Exception as exc:  # surface tool errors to the LLM
                results.append(f"TOOL_CALL: {name} -> ERROR: {exc!r}")
        return results

    @staticmethod
    def _with_tool_results(prompt: str, tool_results: list[str]) -> str:
        """Append tool results to *prompt* so the LLM can answer from them."""
        if not tool_results:
            return prompt
        return (
            f"{prompt}\n\nTool results:\n" + "\n".join(tool_results)
            + "\n\nGiven the tool results above, produce your final answer "
            "(do not call tools again)."
        )

    # ------------------------------------------------------------------
    def can_handle(self, task: Task) -> bool:
        """Return True if this agent is capable of handling *task*.

        Agents without declared capabilities can handle anything; otherwise
        at least one capability must match the task description.
        """
        if not self.capabilities:
            return True
        return any(c.matches(task.description) for c in self.capabilities)

    def handle_task(self, task: Task) -> Message:
        """Convenience: process a :class:`Task` directly."""
        msg = task.to_message(sender="orchestrator")
        msg.recipient = self.id
        return self.process(msg)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<Agent {self.name!r} role={self.role!r} caps={len(self.capabilities)}>"