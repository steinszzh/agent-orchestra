"""Agent Orchestra HTTP API.

A thin FastAPI layer over the framework: submit a task, pick an orchestration
pattern (sequential / parallel / router / hierarchical / debate / workflow),
get back structured JSON with per-agent outputs and timing.

Design notes (interview-ready):
- Structured outputs: Pydantic request/response models — no free-text parsing
  on the API boundary.
- Isolation: a fresh Orchestrator + Memory per request, so concurrent runs
  never share blackboard state.
- Deterministic by default: MockBackend runs offline; flip use_live_model
  (with LLM_API_KEY / LLM_BASE_URL / LLM_MODEL env) to hit any OpenAI-
  compatible provider.
- Errors are mapped to structured 4xx JSON; unknown modes/agents never crash
  the process.

Run:
    uvicorn server.app:app --reload
Test:
    pytest server/tests
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_orchestra import Message, Task, Workflow, WorkflowEngine

from .runtime import (
    DEFAULT_AGENTS,
    DEFAULT_WORKERS,
    build_backend,
    build_orchestrator,
)

Mode = Literal["sequential", "parallel", "router", "hierarchical", "debate", "workflow"]

SUPPORTED_MODES: list[str] = ["sequential", "parallel", "router", "hierarchical", "debate", "workflow"]


class WorkflowStepSpec(BaseModel):
    id: str
    agent: str
    depends_on: list[str] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    steps: list[WorkflowStepSpec]


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    mode: Mode = "sequential"
    agent_ids: Optional[list[str]] = None
    aggregator: Optional[str] = None
    rounds: int = Field(1, ge=1, le=5)
    workflow: Optional[WorkflowSpec] = None
    use_live_model: bool = False


class RunResult(BaseModel):
    run_id: str
    mode: str
    status: str
    results: dict[str, str]
    agent_ids: list[str]
    duration_ms: int


def _normalize(result) -> dict[str, str]:
    """Normalise whatever the orchestrator returns to {agent_id: content}."""
    if isinstance(result, Message):
        return {result.sender: result.content}
    if isinstance(result, dict):
        return {k: v.content if isinstance(v, Message) else str(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return {m.sender: m.content for m in result if isinstance(m, Message)}
    return {"result": str(result)}


def _run_workflow(orch, req: RunRequest) -> dict[str, Message]:
    spec = req.workflow
    if spec is None or not spec.steps:
        raise HTTPException(status_code=422, detail="mode=workflow requires workflow.steps")
    wf = Workflow("api-wf")
    for s in spec.steps:
        wf.step(s.id, s.agent, depends_on=s.depends_on)
    engine = WorkflowEngine(agents=orch.agents, memory=orch.memory)
    try:
        return engine.run(wf, Task(description=req.task))
    finally:
        engine.shutdown()


def _execute(orch, req: RunRequest):
    task = Task(description=req.task)

    if req.mode == "sequential":
        return orch.run_sequential(task, agent_ids=req.agent_ids or DEFAULT_AGENTS)
    if req.mode == "parallel":
        return orch.run_parallel(
            task,
            agent_ids=req.agent_ids or DEFAULT_AGENTS,
            aggregator=req.aggregator,
        )
    if req.mode == "router":
        return orch.run_router(
            task,
            router_agent="router",
            candidate_ids=req.agent_ids or DEFAULT_WORKERS,
        )
    if req.mode == "hierarchical":
        return orch.run_hierarchical(
            task,
            manager_agent="planner",
            worker_ids=req.agent_ids or DEFAULT_WORKERS,
        )
    if req.mode == "debate":
        return orch.run_debate(
            task,
            debater_ids=req.agent_ids or ["analyst", "writer", "researcher"],
            rounds=req.rounds,
            judge_agent="judge",
        )
    if req.mode == "workflow":
        return _run_workflow(orch, req)
    raise HTTPException(status_code=422, detail=f"unsupported mode: {req.mode}")


app = FastAPI(
    title="Agent Orchestra API",
    description="Multi-agent orchestration as a service: sequential / parallel / router / hierarchical / debate / DAG workflows.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    from agent_orchestra import __version__

    return {"status": "ok", "framework_version": __version__, "modes": SUPPORTED_MODES}


@app.get("/v1/modes")
def list_modes() -> dict:
    return {
        "modes": [
            {"name": "sequential", "desc": "pipeline: each agent receives the previous output"},
            {"name": "parallel", "desc": "fan-out: same task to N agents, optional aggregator"},
            {"name": "router", "desc": "a router agent picks the best candidate for the task"},
            {"name": "hierarchical", "desc": "manager decomposes, delegates to workers, synthesises"},
            {"name": "debate", "desc": "N debaters criticise each other, a judge picks the best"},
            {"name": "workflow", "desc": "declarative DAG with fan-in / fan-out (most flexible)"},
        ]
    }


@app.post("/v1/run", response_model=RunResult)
def run(req: RunRequest) -> RunResult:
    orch = build_orchestrator(use_live_model=req.use_live_model)
    t0 = time.perf_counter()
    try:
        raw = _execute(orch, req)
    except (ValueError, KeyError) as exc:  # unknown agent id, bad workflow graph, missing key, ...
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # framework-level failures must not 500-crash silently
        raise HTTPException(status_code=500, detail=f"run failed: {exc}")
    finally:
        orch.shutdown()

    results = _normalize(raw)
    return RunResult(
        run_id=uuid.uuid4().hex,
        mode=req.mode,
        status="ok",
        results=results,
        agent_ids=list(results.keys()),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )
