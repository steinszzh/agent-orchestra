#!/usr/bin/env python3
"""Demonstration of every orchestration pattern in Agent Orchestra.

Each function runs one pattern and prints the results.  Execute the
module directly to see all patterns in action:

    python examples/demo.py

or via the installed console script:

    agent-orchestra-demo
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running the file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_orchestra import (  # noqa: E402
    MockBackend,
    Orchestrator,
    ScriptedBackend,
    Step,
    Task,
    Workflow,
    WorkflowEngine,
    create_default_team,
)


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def show(msg_label: str, messages) -> None:
    """Pretty-print one or more messages."""
    if isinstance(messages, list):
        for i, m in enumerate(messages):
            print(f"\n[{msg_label} step {i}] {m.sender} -> {m.recipient}")
            print(m.content)
    else:
        m = messages
        print(f"\n[{msg_label}] {m.sender} -> {m.recipient}")
        print(m.content)


def demo_sequential() -> None:
    banner("1. SEQUENTIAL PIPELINE  (researcher -> writer -> analyst)")
    orch = Orchestrator("sequential-team")
    orch.register_many(*create_default_team())
    print(orch.summary())

    task = Task("Write a short article about the future of renewable energy.")
    results = orch.run_sequential(task, ["researcher", "writer", "analyst"])
    show("sequential", results)
    print(f"\nTotal messages in memory: {len(orch.memory.history)}")


def demo_parallel() -> None:
    banner("2. PARALLEL FAN-OUT  (researcher + writer + coder, then planner aggregates)")
    orch = Orchestrator("parallel-team")
    orch.register_many(*create_default_team())
    print(orch.summary())

    task = Task("Brainstorm approaches for a smart-home automation system.")
    results = orch.run_parallel(
        task,
        ["researcher", "writer", "coder"],
        aggregator="planner",
    )
    show("parallel", results)


def demo_router() -> None:
    banner("3. ROUTER  (router picks the best agent for each task)")
    orch = Orchestrator("router-team")
    orch.register_many(*create_default_team())
    # Give the router keyword-based rules so it routes sensibly offline.
    # NOTE: avoid bare "code" as a keyword — it matches "coder" in the
    # candidate list that appears in the router prompt.  Use more specific
    # programming-related keywords instead.
    # Use multi-word keywords that won't match the candidate descriptions
    # (e.g. coder's description contains "debugs"/"writes") that also appear
    # in the router prompt.
    orch.agents["router"].llm = MockBackend(rules=[
        ("debug a", "coder"),
        ("python", "coder"),
        ("keyerror", "coder"),
        ("write a", "writer"),
        ("write an", "writer"),
        ("blog", "writer"),
        ("article", "writer"),
        ("analyze the", "analyst"),
        ("analyze this", "analyst"),
        ("research", "researcher"),
        ("find", "researcher"),
    ])
    print(orch.summary())

    candidates = ["researcher", "writer", "coder", "analyst"]
    for desc in [
        "Debug a Python function that throws KeyError.",
        "Write a blog post intro about space exploration.",
        "Analyze the strengths of this marketing copy.",
    ]:
        # Clear memory between tasks so prior routing prompts (which contain
        # keywords like "debug"/"python") don't pollute the next routing call.
        orch.memory.clear()
        print(f"\n--- Task: {desc}")
        result = orch.run_router(
            Task(desc),
            router_agent="router",
            candidate_ids=candidates,
        )
        print(f"Routed to: {result.sender}")
        print(result.content)


def demo_hierarchical() -> None:
    banner("4. HIERARCHICAL  (planner decomposes, delegates, synthesises)")
    orch = Orchestrator("hier-team")
    orch.register_many(*create_default_team())
    print(orch.summary())

    # Use a scripted planner so the JSON plan is parseable deterministically.
    # First response = the plan JSON; second = the synthesised final report.
    orch.agents["planner"].llm = ScriptedBackend(
        responses=[
            '[{"worker": "researcher", "subtask": "Research key trends in AI for 2025."}, '
            '{"worker": "coder", "subtask": "Write a small Python demo of an AI agent."}, '
            '{"worker": "analyst", "subtask": "Review the code and research for quality."}]',
            "Final Report: AI Agents in 2025\n"
            "===============================\n"
            "1. Research Summary: Key trends include tool-use, multi-agent "
            "collaboration, and autonomous planning.\n"
            "2. Code Sample: A minimal agent loop with tool calling.\n"
            "3. Quality Review: Code is clean and research is thorough.\n"
            "Conclusion: AI agents are rapidly maturing toward autonomous, "
            "collaborative systems.",
        ]
    )

    task = Task("Produce a mini-report on AI agents in 2025 with a code sample.")
    final = orch.run_hierarchical(
        task,
        manager_agent="planner",
        worker_ids=["researcher", "coder", "analyst"],
    )
    show("hierarchical-final", final)


def demo_debate() -> None:
    banner("5. DEBATE  (writer vs coder vs analyst, judge decides)")
    orch = Orchestrator("debate-team")
    orch.register_many(*create_default_team())
    print(orch.summary())

    task = Task("What is the best approach to learn programming: projects or tutorials?")
    verdict = orch.run_debate(
        task,
        debater_ids=["writer", "coder", "analyst"],
        rounds=2,
        judge_agent="judge",
    )
    show("debate-verdict", verdict)


def demo_workflow_dag() -> None:
    banner("6. WORKFLOW DAG  (research -> [write || code] -> analyst review)")
    team = {a.id: a for a in create_default_team()}
    engine = WorkflowEngine(team)

    wf = (
        Workflow("report-pipeline")
        .step("research", "researcher", description="gather facts")
        .step("write", "writer", depends_on=["research"], description="draft article")
        .step("code", "coder", depends_on=["research"], description="write demo code")
        .step("review", "analyst", depends_on=["write", "code"], description="review both")
    )

    task = Task("Create a short guide on Python decorators with examples.")
    results = engine.run(wf, task)

    for step_id in ["research", "write", "code", "review"]:
        m = results[step_id]
        print(f"\n[workflow:{step_id}] {m.sender}")
        print(m.content)


def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          AGENT ORCHESTRA — Multi-Agent Orchestration         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    demo_sequential()
    demo_parallel()
    demo_router()
    demo_hierarchical()
    demo_debate()
    demo_workflow_dag()

    banner("DONE — all patterns executed successfully ✅")


if __name__ == "__main__":
    main()