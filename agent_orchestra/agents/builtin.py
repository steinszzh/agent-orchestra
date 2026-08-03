"""Built-in, ready-to-use agent implementations.

These are thin specialisations of :class:`Agent` with sensible system prompts
and capability declarations.  They use the :class:`MockBackend` by default so
the whole framework runs offline, but any :class:`LLMBackend` can be passed.
"""
from __future__ import annotations

from typing import Optional

from ..core.agent import Agent, AgentCapability
from ..core.llm import LLMBackend, MockBackend


class ResearcherAgent(Agent):
    """Gathers information and summarises findings."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Researcher",
            role="research analyst",
            description="Investigates topics, gathers facts, and summarises findings.",
            system_prompt=(
                "You are a meticulous research analyst. You investigate topics, "
                "gather relevant facts from multiple angles, and produce concise, "
                "well-structured summaries with citations of reasoning."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("research", "investigate and summarise", ["research", "find", "investigat", "gather"]),
                AgentCapability("summarize", "condense information", ["summar", "condense", "overview"]),
            ],
            **kw,
        )


class WriterAgent(Agent):
    """Produces polished written content."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Writer",
            role="content writer",
            description="Drafts clear, engaging, well-structured written content.",
            system_prompt=(
                "You are an expert writer. You produce clear, engaging, and "
                "well-structured prose tailored to the audience. You always "
                "include an introduction, body, and conclusion."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("write", "draft content", ["writ", "draft", "compos", "article", "blog", "email"]),
                AgentCapability("edit", "refine prose", ["edit", "refin", "polish", "rewrite"]),
            ],
            **kw,
        )


class AnalystAgent(Agent):
    """Critically evaluates inputs and provides structured analysis."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Analyst",
            role="critical analyst",
            description="Reviews work, identifies strengths/weaknesses, and gives verdicts.",
            system_prompt=(
                "You are a sharp critical analyst. You review content rigorously, "
                "identify strengths and weaknesses, and deliver a clear verdict "
                "with actionable recommendations."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("analyze", "evaluate and critique", ["analy", "review", "critique", "evaluat", "assess"]),
                AgentCapability("verify", "fact-check", ["verif", "fact", "check", "validate"]),
            ],
            **kw,
        )


class CoderAgent(Agent):
    """Generates and reviews code."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Coder",
            role="software engineer",
            description="Writes, reviews, and debugs code across languages.",
            system_prompt=(
                "You are a senior software engineer. You write clean, idiomatic, "
                "well-tested code and explain your design choices. You prefer "
                "simple, readable solutions."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("code", "write code", ["cod", "program", "implement", "function", "class", "bug", "debug"]),
                AgentCapability("review_code", "review code", ["review", "refactor", "optim"]),
            ],
            **kw,
        )


class PlannerAgent(Agent):
    """Decomposes tasks and coordinates other agents."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Planner",
            role="project planner / manager",
            description="Decomposes complex tasks into subtasks and coordinates execution.",
            system_prompt=(
                "You are a strategic project planner. You decompose complex tasks "
                "into clear, ordered subtasks, assign them to the best-suited "
                "agents, and track dependencies. You think step by step."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("plan", "decompose and schedule", ["plan", "decompos", "schedul", "coordinat", "manag"]),
                AgentCapability("route", "assign tasks", ["rout", "assign", "delegat"]),
            ],
            **kw,
        )


class RouterAgent(Agent):
    """Inspects a task and selects the best candidate agent."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Router",
            role="task router",
            description="Analyzes a task and routes it to the most capable agent.",
            system_prompt=(
                "You are a task router. You analyze incoming tasks and select "
                "the single best agent to handle them based on each agent's "
                "capabilities. You reply with only the agent id."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("route", "select best agent", ["rout", "select", "choose", "assign"]),
            ],
            **kw,
        )


class JudgeAgent(Agent):
    """Evaluates multiple answers and selects the best."""

    def __init__(self, llm: Optional[LLMBackend] = None, **kw) -> None:
        super().__init__(
            name="Judge",
            role="adjudicator",
            description="Compares multiple answers and selects the best with reasoning.",
            system_prompt=(
                "You are an impartial judge. You compare candidate answers, "
                "evaluate them against criteria, and select the best one with "
                "clear reasoning."
            ),
            llm=llm or MockBackend(),
            capabilities=[
                AgentCapability("judge", "pick best answer", ["judg", "pick", "select", "best", "verdict"]),
            ],
            **kw,
        )


# Convenience factory ---------------------------------------------------------
def create_default_team(llm: Optional[LLMBackend] = None) -> list[Agent]:
    """Return a balanced team of built-in agents sharing one LLM backend."""
    backend = llm or MockBackend()
    return [
        ResearcherAgent(llm=backend),
        WriterAgent(llm=backend),
        AnalystAgent(llm=backend),
        CoderAgent(llm=backend),
        PlannerAgent(llm=backend),
        RouterAgent(llm=backend),
        JudgeAgent(llm=backend),
    ]