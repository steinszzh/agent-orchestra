"""Unit tests for Agent Orchestra.

Run with::

    python -m pytest tests/ -v

or without pytest::

    python tests/test_orchestra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_orchestra import (  # noqa: E402
    Agent,
    AgentCapability,
    Memory,
    Message,
    MessageRole,
    MockBackend,
    OpenAIBackend,
    Orchestrator,
    ScriptedBackend,
    Step,
    Task,
    Tool,
    Workflow,
    WorkflowEngine,
    create_default_team,
)


# ---------------------------------------------------------------------------
# Message & Task
# ---------------------------------------------------------------------------
class TestMessage:
    def test_message_defaults(self):
        m = Message(content="hello")
        assert m.role == MessageRole.ASSISTANT
        assert m.sender == "system"
        assert m.recipient is None
        assert m.id  # uuid generated

    def test_to_dict(self):
        m = Message(content="hi", sender="a", recipient="b")
        d = m.to_dict()
        assert d["content"] == "hi"
        assert d["sender"] == "a"
        assert d["recipient"] == "b"
        assert d["role"] == "assistant"

    def test_task_to_message(self):
        t = Task(description="do work", assigned_to="writer")
        m = t.to_message(sender="orch")
        assert m.content == "do work"
        assert m.recipient == "writer"
        assert m.sender == "orch"
        assert m.role == MessageRole.USER
        assert m.metadata["task_id"] == t.id


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class TestMemory:
    def test_set_get(self):
        mem = Memory()
        mem.set("k", "v")
        assert mem.get("k") == "v"
        assert mem["k"] == "v"
        assert "k" in mem

    def test_append(self):
        mem = Memory()
        mem.append("list", 1)
        mem.append("list", 2)
        assert mem.get("list") == [1, 2]

    def test_history(self):
        mem = Memory()
        m1 = Message(content="a")
        m2 = Message(content="b")
        mem.add_message(m1)
        mem.add_message(m2)
        assert len(mem.history) == 2
        assert mem.recent(1) == [m2]

    def test_clear(self):
        mem = Memory()
        mem.set("k", "v")
        mem.add_message(Message(content="x"))
        mem.clear()
        assert mem.get("k") is None
        assert len(mem.history) == 0

    def test_snapshot_is_copy(self):
        mem = Memory()
        mem.set("k", "v")
        snap = mem.snapshot()
        snap["k"] = "changed"
        assert mem.get("k") == "v"


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
class TestBackends:
    def test_mock_role_aware_researcher(self):
        b = MockBackend()
        out = b.generate("find info", system_prompt="You are a research analyst.")
        assert "Research findings" in out

    def test_mock_role_aware_coder(self):
        b = MockBackend()
        out = b.generate("write a function", system_prompt="You are a software engineer.")
        assert "```python" in out

    def test_mock_rules(self):
        b = MockBackend(rules=[("weather", "It's sunny: {prompt}")])
        out = b.generate("what's the weather")
        assert "sunny" in out

    def test_mock_scripted_priority(self):
        b = MockBackend(responses=["first", "second"])
        assert b.generate("anything") == "first"
        assert b.generate("anything") == "second"
        # falls back to role-aware after scripted exhausted
        assert "MockLLM" not in b.generate("x", system_prompt="You are a research analyst.")

    def test_scripted_backend_cycles(self):
        b = ScriptedBackend(["a", "b"])
        assert b.generate("x") == "a"
        assert b.generate("x") == "b"
        assert b.generate("x") == "a"  # cycles


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class TestAgent:
    def test_agent_id_from_name(self):
        a = Agent(name="My Agent", role="test")
        assert a.id == "my_agent"

    def test_agent_process(self):
        a = Agent(name="Test", role="tester", llm=ScriptedBackend(["ok"]))
        msg = Message(content="hello", sender="user", recipient="test")
        resp = a.process(msg)
        assert resp.content == "ok"
        assert resp.sender == "test"
        assert resp.recipient == "user"

    def test_capability_matches(self):
        cap = AgentCapability("code", "write code", ["python", "function"])
        assert cap.matches("write a python function")
        assert not cap.matches("write a poem")

    def test_default_team(self):
        team = create_default_team()
        assert len(team) == 7
        ids = {a.id for a in team}
        assert {"researcher", "writer", "analyst", "coder", "planner", "router", "judge"} == ids


# ---------------------------------------------------------------------------
# Orchestrator patterns
# ---------------------------------------------------------------------------
class TestOrchestrator:
    def _make_orch(self):
        orch = Orchestrator("test")
        orch.register_many(*create_default_team())
        return orch

    def test_register_and_get(self):
        orch = self._make_orch()
        assert orch.get("researcher") is not None
        assert orch.get("nonexistent") is None
        assert len(orch.list_agents()) == 7

    def test_send_requires_recipient(self):
        orch = self._make_orch()
        with pytest.raises(ValueError):
            orch.send(Message(content="hi"))

    def test_send_unknown_agent(self):
        orch = self._make_orch()
        with pytest.raises(KeyError):
            orch.send(Message(content="hi", recipient="ghost"))

    def test_dispatch(self):
        orch = self._make_orch()
        resp = orch.dispatch(Task("say hi", assigned_to="writer"))
        assert resp.sender == "writer"

    def test_sequential(self):
        orch = self._make_orch()
        results = orch.run_sequential(
            Task("write something"),
            ["researcher", "writer"],
        )
        assert len(results) == 2
        assert results[0].sender == "researcher"
        assert results[1].sender == "writer"

    def test_parallel(self):
        orch = self._make_orch()
        results = orch.run_parallel(
            Task("brainstorm"),
            ["researcher", "writer", "coder"],
        )
        assert len(results) == 3

    def test_parallel_with_aggregator(self):
        orch = self._make_orch()
        results = orch.run_parallel(
            Task("brainstorm"),
            ["researcher", "writer"],
            aggregator="planner",
        )
        assert len(results) == 3  # 2 + aggregator
        assert results[-1].sender == "planner"

    def test_router(self):
        orch = self._make_orch()
        # Script the router to pick "coder".
        orch.agents["router"].llm = ScriptedBackend(["coder"])
        resp = orch.run_router(
            Task("debug a function"),
            router_agent="router",
            candidate_ids=["researcher", "writer", "coder"],
        )
        assert resp.sender == "coder"

    def test_router_fallback(self):
        orch = self._make_orch()
        # Router gives a non-matching answer -> fallback to can_handle.
        orch.agents["router"].llm = ScriptedBackend(["I don't know"])
        resp = orch.run_router(
            Task("debug a python function"),
            router_agent="router",
            candidate_ids=["researcher", "writer", "coder"],
        )
        # Should still dispatch to someone.
        assert resp.sender in {"researcher", "writer", "coder"}

    def test_hierarchical(self):
        orch = self._make_orch()
        orch.agents["planner"].llm = ScriptedBackend(
            responses=[
                '[{"worker": "researcher", "subtask": "research"}, '
                '{"worker": "coder", "subtask": "code"}]'
            ]
        )
        final = orch.run_hierarchical(
            Task("build a report"),
            manager_agent="planner",
            worker_ids=["researcher", "coder"],
        )
        assert final.sender == "planner"

    def test_debate(self):
        orch = self._make_orch()
        verdict = orch.run_debate(
            Task("best language?"),
            debater_ids=["writer", "coder"],
            rounds=2,
            judge_agent="judge",
        )
        assert verdict.sender == "judge"

    def test_memory_shared(self):
        orch = self._make_orch()
        orch.run_sequential(Task("x"), ["researcher", "writer"])
        # All agents share the orchestrator's memory.
        assert orch.agents["researcher"].memory is orch.memory
        assert len(orch.memory.history) > 0

    def test_summary(self):
        orch = self._make_orch()
        s = orch.summary()
        assert "test" in s
        assert "researcher" in s


# ---------------------------------------------------------------------------
# Workflow / DAG
# ---------------------------------------------------------------------------
class TestWorkflow:
    def test_add_step_and_validate(self):
        wf = Workflow("t")
        wf.step("a", "researcher")
        wf.step("b", "writer", depends_on=["a"])
        wf.validate()
        assert set(wf.steps) == {"a", "b"}

    def test_duplicate_step_raises(self):
        wf = Workflow("t")
        wf.step("a", "researcher")
        with pytest.raises(ValueError):
            wf.step("a", "writer")

    def test_missing_dep_raises(self):
        wf = Workflow("t")
        with pytest.raises(ValueError):
            wf.step("b", "writer", depends_on=["a"])

    def test_cycle_detection(self):
        wf = Workflow("t")
        wf.step("a", "researcher", depends_on=[])
        # 'a' exists now, add 'b' depending on 'a', then make 'a' depend on 'b'
        wf.step("b", "writer", depends_on=["a"])
        wf.steps["a"].depends_on = ["b"]  # create a cycle
        with pytest.raises(ValueError, match="Cycle"):
            wf.validate()

    def test_engine_runs_dag(self):
        team = {a.id: a for a in create_default_team()}
        engine = WorkflowEngine(team)
        wf = (
            Workflow("pipe")
            .step("research", "researcher")
            .step("write", "writer", depends_on=["research"])
            .step("review", "analyst", depends_on=["write"])
        )
        results = engine.run(wf, Task("make a guide"))
        assert set(results) == {"research", "write", "review"}
        assert results["review"].sender == "analyst"

    def test_engine_parallel_branches(self):
        team = {a.id: a for a in create_default_team()}
        engine = WorkflowEngine(team)
        wf = (
            Workflow("fan")
            .step("r", "researcher")
            .step("w", "writer", depends_on=["r"])
            .step("c", "coder", depends_on=["r"])
            .step("a", "analyst", depends_on=["w", "c"])
        )
        results = engine.run(wf, Task("project"))
        assert results["a"].sender == "analyst"

    def test_engine_missing_agent_raises(self):
        engine = WorkflowEngine({})
        wf = Workflow("t").step("a", "researcher")
        with pytest.raises(KeyError):
            engine.run(wf, Task("x"))


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------
class TestToolCalling:
    def test_tool_executed_and_result_fed_back(self):
        calls: list[str] = []

        def echo(text: str) -> str:
            calls.append(text)
            return f"echo:{text}"

        agent = Agent(
            name="Assistant",
            role="assistant",
            tools=[
                Tool(
                    name="echo",
                    description="echo text",
                    function=echo,
                    parameters={"text": {"type": "string"}},
                )
            ],
            llm=ScriptedBackend(
                [
                    'TOOL_CALL: echo({"text": "hello tool"})',
                    "The tool said: hello tool",
                ]
            ),
        )
        resp = agent.process(Message(content="use the tool", sender="user"))
        assert calls == ["hello tool"]
        assert "hello tool" in resp.content

    def test_no_tool_call_returns_directly(self):
        agent = Agent(
            name="Assistant",
            role="assistant",
            llm=ScriptedBackend(["plain answer"]),
        )
        resp = agent.process(Message(content="hi", sender="user"))
        assert resp.content == "plain answer"

    def test_parse_tool_calls(self):
        assert Agent._parse_tool_calls('TOOL_CALL: foo({"a": 1})') == [("foo", {"a": 1})]
        assert Agent._parse_tool_calls("no calls here") == []
        assert Agent._parse_tool_calls('TOOL_CALL: foo({"broken")') == []
        assert Agent._parse_tool_calls('TOOL_CALL: foo({"a": "b"}) TOOL_CALL: bar({"c": 2})') == [
            ("foo", {"a": "b"}),
            ("bar", {"c": 2}),
        ]

    def test_unknown_tool_reported(self):
        results = Agent(
            name="Assistant", role="assistant"
        )._execute_tools([("nope", {"a": 1})])
        assert "unknown tool" in results[0]

    def test_tool_error_surfaced(self):
        def boom() -> str:
            raise ValueError("bad thing")

        agent = Agent(
            name="Assistant",
            role="assistant",
            tools=[Tool(name="boom", description="boom", function=boom)],
        )
        results = agent._execute_tools([("boom", {})])
        assert "ERROR" in results[0] and "bad thing" in results[0]

    def test_max_tool_rounds_capped(self):
        backend = ScriptedBackend(['TOOL_CALL: echo({"text": "x"})'] * 10)
        agent = Agent(
            name="Assistant",
            role="assistant",
            max_tool_rounds=2,
            tools=[Tool(name="echo", description="echo", function=lambda text: text)],
            llm=backend,
        )
        agent.process(Message(content="x", sender="user"))
        # Loop runs max_tool_rounds+1 = 3 times regardless of scripted backlog.
        assert backend._idx == 3

    def test_tools_advertised_in_default_system_prompt(self):
        agent = Agent(
            name="Assistant",
            role="assistant",
            tools=[Tool(name="echo", description="echo text", function=lambda text: text)],
        )
        assert "echo" in agent.system_prompt
        assert "TOOL_CALL" in agent.system_prompt


# ---------------------------------------------------------------------------
# Capability routing (regression for the `or True` bug)
# ---------------------------------------------------------------------------
class TestCapabilityRouting:
    def test_can_handle_respects_capabilities(self):
        agent = Agent(
            name="Coder",
            role="software engineer",
            capabilities=[AgentCapability("code", "write code", ["python", "debug"])],
        )
        assert agent.can_handle(Task("debug a python bug"))
        assert not agent.can_handle(Task("write a blog post"))

    def test_no_capabilities_handles_anything(self):
        agent = Agent(name="Plain", role="assistant")
        assert agent.can_handle(Task("anything"))

    def test_router_fallback_respects_capabilities(self):
        # Regression: can_handle used to return True always, so the router
        # fallback always picked the first candidate. Now a task that only
        # the coder can handle must route to the coder.
        orch = Orchestrator()
        orch.register_many(
            Agent(
                "Writer",
                "content writer",
                capabilities=[AgentCapability("write", "draft", ["write", "blog"])],
            ),
            Agent(
                "Coder",
                "software engineer",
                capabilities=[AgentCapability("code", "write code", ["python", "debug"])],
            ),
        )
        orch.register(Agent("Router", "task router", llm=ScriptedBackend(["garbage output"])))
        msg = orch.run_router(
            Task("debug a python function"),
            router_agent="router",
            candidate_ids=["writer", "coder"],
        )
        assert msg.sender == "coder"


# ---------------------------------------------------------------------------
# OpenAIBackend
# ---------------------------------------------------------------------------
class TestOpenAIBackend:
    def test_constructs_without_openai_installed(self):
        backend = OpenAIBackend(
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        assert backend.model == "deepseek-chat"
        assert backend.base_url == "https://api.deepseek.com/v1"

    def test_missing_openai_raises_clear_error(self, monkeypatch):
        backend = OpenAIBackend(api_key="sk-test")

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(ImportError, match=r"agent-orchestra\[openai\]"):
            backend.generate("hi")


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))