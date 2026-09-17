"""Agent Orchestra API tests.

Deterministic by design: MockBackend means no API key / network needed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "sequential" in body["modes"]


def test_list_modes():
    r = client.get("/v1/modes")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["modes"]]
    assert names == ["sequential", "parallel", "router", "hierarchical", "debate", "workflow"]


def test_run_sequential_returns_structured_results():
    r = client.post(
        "/v1/run",
        json={
            "task": "Summarise the key risks of scaling hydrogen refueling.",
            "mode": "sequential",
            "agent_ids": ["researcher", "writer"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "sequential"
    assert set(body["results"].keys()) == {"researcher", "writer"}
    assert all(v for v in body["results"].values())
    assert body["duration_ms"] >= 0


def test_run_parallel_with_aggregator():
    r = client.post(
        "/v1/run",
        json={
            "task": "Propose three maintenance strategies.",
            "mode": "parallel",
            "agent_ids": ["researcher", "analyst"],
            "aggregator": "writer",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"].keys()) == {"researcher", "analyst", "writer"}


def test_run_router_falls_back_gracefully():
    r = client.post(
        "/v1/run",
        json={
            "task": "Debug this python snippet: print('hi'",
            "mode": "router",
            "agent_ids": ["coder", "analyst"],
        },
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1


def test_run_hierarchical():
    r = client.post(
        "/v1/run",
        json={
            "task": "Plan and draft a quarterly data-quality report.",
            "mode": "hierarchical",
            "agent_ids": ["researcher", "writer"],
        },
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "hierarchical"


def test_run_debate_with_judge():
    r = client.post(
        "/v1/run",
        json={
            "task": "Is RAG better than fine-tuning for a 500-doc internal KB?",
            "mode": "debate",
            "agent_ids": ["analyst", "researcher"],
            "rounds": 1,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "judge" in body["results"]


def test_run_workflow_dag():
    r = client.post(
        "/v1/run",
        json={
            "task": "Write a two-section briefing on battery safety.",
            "mode": "workflow",
            "workflow": {
                "steps": [
                    {"id": "research", "agent": "researcher"},
                    {"id": "write", "agent": "writer", "depends_on": ["research"]},
                    {"id": "code", "agent": "coder", "depends_on": ["research"]},
                    {"id": "review", "agent": "analyst", "depends_on": ["write", "code"]},
                ]
            },
        },
    )
    assert r.status_code == 200
    assert set(r.json()["results"].keys()) == {"research", "write", "code", "review"}


def test_run_unknown_mode_returns_422():
    r = client.post("/v1/run", json={"task": "hello", "mode": "teleport"})
    assert r.status_code == 422


def test_run_unknown_agent_returns_400():
    r = client.post(
        "/v1/run",
        json={"task": "hello", "mode": "sequential", "agent_ids": ["nobody"]},
    )
    assert r.status_code == 400


def test_run_empty_task_rejected():
    r = client.post("/v1/run", json={"task": "", "mode": "sequential"})
    assert r.status_code == 422


def test_workflow_mode_requires_spec():
    r = client.post("/v1/run", json={"task": "hello", "mode": "workflow"})
    assert r.status_code == 422


def test_workflow_missing_dependency_rejected():
    r = client.post(
        "/v1/run",
        json={
            "task": "hello",
            "mode": "workflow",
            "workflow": {"steps": [{"id": "a", "agent": "researcher", "depends_on": ["ghost"]}]},
        },
    )
    assert r.status_code == 400
