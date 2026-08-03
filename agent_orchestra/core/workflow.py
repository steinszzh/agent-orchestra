"""Declarative workflow / DAG orchestration.

A :class:`Workflow` is a directed acyclic graph of steps.  Each step maps to
an agent and depends on zero or more upstream steps.  The engine runs steps
as soon as all their dependencies have completed, passing the collected
upstream outputs as context.  This is the most flexible pattern and can
express sequential, parallel, and fan-in/fan-out topologies.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from .agent import Agent
from .memory import Memory
from .message import Message, MessageRole, Task


@dataclass
class Step:
    """A single node in the workflow DAG."""

    id: str
    agent_id: str
    depends_on: list[str] = field(default_factory=list)
    prompt_template: str = "{task}\n\nUpstream context:\n{upstream}"
    description: str = ""

    def __post_init__(self) -> None:
        if self.id in self.depends_on:
            raise ValueError(f"Step {self.id!r} cannot depend on itself.")


class Workflow:
    """A DAG of :class:`Step` objects executed by :class:`WorkflowEngine`."""

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self.steps: dict[str, Step] = {}

    def add_step(self, step: Step) -> Step:
        if step.id in self.steps:
            raise ValueError(f"Duplicate step id: {step.id!r}")
        for dep in step.depends_on:
            if dep not in self.steps:
                raise ValueError(
                    f"Step {step.id!r} depends on unknown step {dep!r}. "
                    f"Add {dep!r} first."
                )
        self.steps[step.id] = step
        return step

    def step(
        self,
        id: str,
        agent_id: str,
        *,
        depends_on: Optional[list[str]] = None,
        prompt_template: str = "{task}\n\nUpstream context:\n{upstream}",
        description: str = "",
    ) -> Workflow:
        """Fluent builder for adding a step."""
        self.add_step(
            Step(
                id=id,
                agent_id=agent_id,
                depends_on=depends_on or [],
                prompt_template=prompt_template,
                description=description,
            )
        )
        return self

    def roots(self) -> list[str]:
        return [sid for sid, s in self.steps.items() if not s.depends_on]

    def validate(self) -> None:
        """Check the graph is a valid DAG (no cycles, deps exist)."""
        for s in self.steps.values():
            for dep in s.depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Step {s.id!r} depends on missing step {dep!r}.")
        # Cycle detection via DFS.
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {sid: WHITE for sid in self.steps}

        def dfs(sid: str) -> None:
            color[sid] = GRAY
            for dep in self.steps[sid].depends_on:
                if color[dep] == GRAY:
                    raise ValueError(f"Cycle detected involving {sid!r} -> {dep!r}.")
                if color[dep] == WHITE:
                    dfs(dep)
            color[sid] = BLACK

        for sid in self.steps:
            if color[sid] == WHITE:
                dfs(sid)


class WorkflowEngine:
    """Executes a :class:`Workflow` against a set of registered agents."""

    def __init__(
        self,
        agents: dict[str, Agent],
        *,
        memory: Optional[Memory] = None,
        max_workers: int = 8,
    ) -> None:
        """Initialise the WorkflowEngine.

        Args:
            agents: A mapping of agent id to :class:`Agent` instance.
                All agents referenced in the workflow must be present.
            memory: A shared :class:`Memory` instance for workflow
                execution.  A new :class:`Memory` is created when
                ``None``.
            max_workers: Maximum number of threads for concurrent
                step execution.
        """
        self.agents = agents
        self.memory = memory or Memory()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(self, workflow: Workflow, task: Task) -> dict[str, Message]:
        """Execute *workflow* for *task*; return a map of step_id -> Message."""
        workflow.validate()
        results: dict[str, Message] = {}
        remaining = set(workflow.steps)

        while remaining:
            # Find steps whose deps are all satisfied.
            ready = [
                sid for sid in remaining
                if all(dep in results for dep in workflow.steps[sid].depends_on)
            ]
            if not ready:
                raise RuntimeError("Workflow deadlock: no runnable steps (cycle?).")

            futures = {}
            for sid in ready:
                step = workflow.steps[sid]
                agent = self.agents.get(step.agent_id)
                if agent is None:
                    raise KeyError(
                        f"Step {sid!r} requires agent {step.agent_id!r} which is not registered."
                    )
                upstream = "\n".join(
                    f"[{dep}]: {results[dep].content}" for dep in step.depends_on
                ) or "(none)"
                content = step.prompt_template.format(
                    task=task.description,
                    upstream=upstream,
                )
                msg = Message(
                    content=content,
                    role=MessageRole.USER,
                    sender="workflow_engine",
                    recipient=step.agent_id,
                    metadata={"task_id": task.id, "step": sid, "workflow": workflow.name},
                )
                self.memory.add_message(msg)
                fut = self._executor.submit(agent.process, msg)
                futures[sid] = fut

            for sid, fut in futures.items():
                resp = fut.result()
                self.memory.add_message(resp)
                results[sid] = resp
                remaining.discard(sid)

        return results

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)