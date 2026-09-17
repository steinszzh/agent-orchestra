"""Runtime factory for the Agent Orchestra HTTP API.

A fresh Orchestrator + Memory is created per request so concurrent runs are
isolated (the framework's shared-blackboard Memory is intentionally scoped to
a single run at the API boundary).

Model selection:
- Default: MockBackend — deterministic, offline, no API key (used by tests).
- Live:    set LLM_API_KEY (+ optional LLM_BASE_URL / LLM_MODEL) and pass
           use_live_model=true. Any OpenAI-compatible endpoint works
           (DeepSeek, SiliconFlow, Ollama OpenAI-mode, ...).
"""
from __future__ import annotations

import os

from agent_orchestra import (
    Agent,
    MockBackend,
    OpenAIBackend,
    Orchestrator,
    create_default_team,
)

DEFAULT_AGENTS = ["researcher", "writer", "analyst"]
DEFAULT_WORKERS = ["researcher", "coder", "analyst"]


def build_backend(use_live_model: bool):
    if not use_live_model:
        return MockBackend()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("use_live_model=true requires LLM_API_KEY env var")
    return OpenAIBackend(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        timeout=float(os.environ.get("LLM_TIMEOUT", "60")),
    )


def build_orchestrator(use_live_model: bool = False) -> Orchestrator:
    """Build a per-request orchestrator with the default 7-agent team."""
    backend = build_backend(use_live_model)
    orch = Orchestrator(f"api-{os.urandom(4).hex()}")
    orch.register_many(*create_default_team(backend))
    return orch


def attach_backend(orch: Orchestrator, backend) -> None:
    """Point every registered agent at the same backend (post-registration)."""
    for agent in orch.agents.values():
        agent.llm = backend
