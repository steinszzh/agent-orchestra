"""Agent Orchestra — a lightweight multi-agent orchestration framework.

A self-contained, dependency-free framework for coordinating multiple agents
through patterns like sequential pipelines, parallel fan-out, routing,
hierarchical delegation, debate, and declarative DAG workflows.

Quick start::

    from agent_orchestra import Orchestrator, Task, create_default_team

    orch = Orchestrator("my-team")
    orch.register_many(*create_default_team())
    result = orch.run_sequential(
        Task("Write a short article about renewable energy."),
        agent_ids=["researcher", "writer", "analyst"],
    )
    print(result[-1].content)
"""
from __future__ import annotations

from .core import (
    Agent,
    AgentCapability,
    LLMBackend,
    Memory,
    Message,
    MessageRole,
    MockBackend,
    Orchestrator,
    ScriptedBackend,
    Step,
    Task,
    Workflow,
    WorkflowEngine,
)
from .agents import (
    AnalystAgent,
    CoderAgent,
    JudgeAgent,
    PlannerAgent,
    ResearcherAgent,
    RouterAgent,
    WriterAgent,
    create_default_team,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "AgentCapability",
    "LLMBackend",
    "Memory",
    "Message",
    "MessageRole",
    "MockBackend",
    "Orchestrator",
    "ScriptedBackend",
    "Step",
    "Task",
    "Workflow",
    "WorkflowEngine",
    # Built-in agents
    "AnalystAgent",
    "CoderAgent",
    "JudgeAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "RouterAgent",
    "WriterAgent",
    "create_default_team",
    "__version__",
]