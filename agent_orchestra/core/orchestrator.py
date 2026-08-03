"""Orchestrator: coordinates multiple agents through different patterns."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .agent import Agent
from .llm import LLMBackend, MockBackend
from .memory import Memory
from .message import Message, MessageRole, Task


class Orchestrator:
    """Central coordinator that manages a registry of agents.

    Supports several orchestration patterns:

    * **sequential** – agents run one after another, each receiving the
      previous agent's output.
    * **parallel** – agents run concurrently on the same input; results are
      aggregated.
    * **router** – a router agent inspects the task and picks the best
      candidate agent to handle it.
    * **hierarchical** – a manager agent decomposes the task and delegates
      subtasks to worker agents, then synthesises a final answer.
    * **debate** – multiple agents discuss the task in rounds and a judge
      agent picks the best answer.
    """

    def __init__(
        self,
        name: str = "orchestrator",
        *,
        llm: Optional[LLMBackend] = None,
        memory: Optional[Memory] = None,
        max_workers: int = 8,
    ) -> None:
        """Initialise the Orchestrator.

        Args:
            name: A human-readable name for this orchestrator instance.
            llm: The default LLM backend for all agents.
                Defaults to :class:`MockBackend` when none is provided.
            memory: A shared :class:`Memory` instance for all agents.
                A new :class:`Memory` is created when ``None``.
            max_workers: Maximum number of threads for parallel
                execution.
        """
        self.name = name
        self.id = name.lower().replace(" ", "_")
        self.llm = llm or MockBackend()
        self.memory = memory or Memory()
        self.agents: dict[str, Agent] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def register(self, agent: Agent) -> Agent:
        """Register an agent and wire it to shared memory."""
        agent.memory = self.memory  # share the blackboard
        self.agents[agent.id] = agent
        return agent

    def register_many(self, *agents: Agent) -> list[Agent]:
        return [self.register(a) for a in agents]

    def get(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        return list(self.agents.values())

    # ------------------------------------------------------------------
    # Low-level dispatch
    # ------------------------------------------------------------------
    def send(self, message: Message) -> Message:
        """Route *message* to its recipient agent and return the response."""
        if not message.recipient:
            raise ValueError("Message must have a recipient to be sent.")
        agent = self.agents.get(message.recipient)
        if agent is None:
            raise KeyError(f"No agent registered with id={message.recipient!r}")
        self.memory.add_message(message)
        response = agent.process(message)
        self.memory.add_message(response)
        return response

    def dispatch(self, task: Task) -> Message:
        """Send a task to its assigned agent (``task.assigned_to``)."""
        if not task.assigned_to:
            raise ValueError("Task.assigned_to must be set to dispatch it.")
        agent = self.agents[task.assigned_to]
        self.memory.add_message(task.to_message(sender=self.id))
        response = agent.handle_task(task)
        self.memory.add_message(response)
        return response

    # ------------------------------------------------------------------
    # Pattern: sequential pipeline
    # ------------------------------------------------------------------
    def run_sequential(
        self,
        task: Task,
        agent_ids: list[str],
        *,
        pass_output: bool = True,
    ) -> list[Message]:
        """Run agents in order, piping each output into the next agent.

        Args:
            task: The initial task.
            agent_ids: Ordered list of agent ids to run.
            pass_output: If True, each agent receives the previous agent's
                output appended to the original task description.
        """
        results: list[Message] = []
        current = task.description
        for i, aid in enumerate(agent_ids):
            agent = self._require(aid)
            content = current if (pass_output and i > 0) else task.description
            if pass_output and i > 0:
                content = (
                    f"Previous agent output:\n{current}\n\n"
                    f"Continue the work on the original task: {task.description}"
                )
            msg = Message(
                content=content,
                role=MessageRole.USER,
                sender=self.id,
                recipient=aid,
                metadata={"task_id": task.id, "step": i, "pattern": "sequential"},
            )
            self.memory.add_message(msg)
            resp = agent.process(msg)
            self.memory.add_message(resp)
            results.append(resp)
            current = resp.content
        return results

    # ------------------------------------------------------------------
    # Pattern: parallel fan-out
    # ------------------------------------------------------------------
    def run_parallel(
        self,
        task: Task,
        agent_ids: list[str],
        *,
        aggregator: Optional[str] = None,
    ) -> list[Message]:
        """Run multiple agents concurrently on the same task.

        If *aggregator* (an agent id) is given, its output is appended to
        the results as the synthesised summary.
        """
        futures = []
        for aid in agent_ids:
            agent = self._require(aid)
            msg = Message(
                content=task.description,
                role=MessageRole.USER,
                sender=self.id,
                recipient=aid,
                metadata={"task_id": task.id, "pattern": "parallel"},
            )
            self.memory.add_message(msg)
            fut = self._executor.submit(self._process_agent, agent, msg)
            futures.append((aid, fut))

        results: list[Message] = []
        for aid, fut in futures:
            resp = fut.result()
            self.memory.add_message(resp)
            results.append(resp)

        if aggregator:
            agg_agent = self._require(aggregator)
            combined = "\n\n---\n\n".join(
                f"[{r.sender}]: {r.content}" for r in results
            )
            msg = Message(
                content=(
                    f"Synthesize the following parallel outputs into a single "
                    f"coherent result:\n\n{combined}"
                ),
                role=MessageRole.USER,
                sender=self.id,
                recipient=aggregator,
                metadata={"task_id": task.id, "pattern": "parallel_aggregate"},
            )
            self.memory.add_message(msg)
            resp = agg_agent.process(msg)
            self.memory.add_message(resp)
            results.append(resp)
        return results

    # ------------------------------------------------------------------
    # Pattern: router
    # ------------------------------------------------------------------
    def run_router(
        self,
        task: Task,
        *,
        router_agent: str,
        candidate_ids: list[str],
    ) -> Message:
        """Let a router agent choose which candidate handles *task*."""
        router = self._require(router_agent)
        candidates_desc = "\n".join(
            f"- {aid}: {self.agents[aid].role} — {self.agents[aid].description}"
            for aid in candidate_ids
            if aid in self.agents
        )
        route_prompt = (
            f"Given the task below, choose the single best agent to handle it.\n\n"
            f"Candidates:\n{candidates_desc}\n\n"
            f"Task: {task.description}\n\n"
            f"Reply with ONLY the agent id (one of: {', '.join(candidate_ids)})."
        )
        msg = Message(
            content=route_prompt,
            role=MessageRole.USER,
            sender=self.id,
            recipient=router_agent,
            metadata={"task_id": task.id, "pattern": "router"},
        )
        self.memory.add_message(msg)
        route_resp = router.process(msg)
        self.memory.add_message(route_resp)

        chosen = self._extract_agent_id(route_resp.content, candidate_ids)
        if not chosen:
            # Fallback: pick the first candidate that can_handle the task.
            chosen = next(
                (aid for aid in candidate_ids if self.agents[aid].can_handle(task)),
                candidate_ids[0],
            )
        return self.dispatch(Task(description=task.description, assigned_to=chosen, context=task.context))

    # ------------------------------------------------------------------
    # Pattern: hierarchical delegation
    # ------------------------------------------------------------------
    def run_hierarchical(
        self,
        task: Task,
        *,
        manager_agent: str,
        worker_ids: list[str],
    ) -> Message:
        """A manager decomposes the task, delegates to workers, synthesises."""
        manager = self._require(manager_agent)
        workers_desc = "\n".join(
            f"- {aid}: {self.agents[aid].role}" for aid in worker_ids if aid in self.agents
        )
        plan_prompt = (
            f"You are coordinating a team. Decompose the following task into "
            f"subtasks and assign each to a worker agent.\n\n"
            f"Workers:\n{workers_desc}\n\n"
            f"Task: {task.description}\n\n"
            f"Respond as JSON: a list of objects with keys "
            f'"worker" (agent id) and "subtask" (string).'
        )
        msg = Message(
            content=plan_prompt,
            role=MessageRole.USER,
            sender=self.id,
            recipient=manager_agent,
            metadata={"task_id": task.id, "pattern": "hierarchical_plan"},
        )
        self.memory.add_message(msg)
        plan_resp = manager.process(msg)
        self.memory.add_message(plan_resp)

        subtasks = self._parse_plan(plan_resp.content, worker_ids)

        # Delegate subtasks (sequentially to keep shared memory coherent).
        worker_outputs: list[Message] = []
        for sub in subtasks:
            wid = sub["worker"]
            stask = sub["subtask"]
            if wid not in self.agents:
                continue
            wmsg = Message(
                content=stask,
                role=MessageRole.USER,
                sender=self.id,
                recipient=wid,
                metadata={"task_id": task.id, "pattern": "hierarchical_worker"},
            )
            self.memory.add_message(wmsg)
            wresp = self.agents[wid].process(wmsg)
            self.memory.add_message(wresp)
            worker_outputs.append(wresp)

        # Synthesise.
        combined = "\n\n".join(f"[{r.sender}]: {r.content}" for r in worker_outputs)
        synth_prompt = (
            f"Here are the worker outputs for the original task. "
            f"Synthesize a final, coherent answer.\n\n"
            f"Original task: {task.description}\n\n"
            f"Worker outputs:\n{combined}"
        )
        smsg = Message(
            content=synth_prompt,
            role=MessageRole.USER,
            sender=self.id,
            recipient=manager_agent,
            metadata={"task_id": task.id, "pattern": "hierarchical_synthesize"},
        )
        self.memory.add_message(smsg)
        final = manager.process(smsg)
        self.memory.add_message(final)
        return final

    # ------------------------------------------------------------------
    # Pattern: debate
    # ------------------------------------------------------------------
    def run_debate(
        self,
        task: Task,
        *,
        debater_ids: list[str],
        rounds: int = 2,
        judge_agent: Optional[str] = None,
    ) -> Message:
        """Agents take turns answering, then critique each other for *rounds*.

        A judge agent (or the last debater) picks the best final answer.
        """
        answers: dict[str, list[str]] = {aid: [] for aid in debater_ids}

        for rnd in range(rounds):
            for aid in debater_ids:
                agent = self._require(aid)
                if rnd == 0:
                    content = task.description
                else:
                    others = "\n\n".join(
                        f"[{o}]: {answers[o][-1]}"
                        for o in debater_ids
                        if o != aid and answers[o]
                    )
                    content = (
                        f"Here are other agents' answers:\n{others}\n\n"
                        f"Critique them and provide your improved answer to: {task.description}"
                    )
                msg = Message(
                    content=content,
                    role=MessageRole.USER,
                    sender=self.id,
                    recipient=aid,
                    metadata={"task_id": task.id, "pattern": "debate", "round": rnd},
                )
                self.memory.add_message(msg)
                resp = agent.process(msg)
                self.memory.add_message(resp)
                answers[aid].append(resp.content)

        # Judge.
        all_answers = "\n\n".join(
            f"[{aid}] (final): {answers[aid][-1]}" for aid in debater_ids
        )
        judge_id = judge_agent or debater_ids[-1]
        judge = self._require(judge_id)
        judge_prompt = (
            f"Several agents debated this task. Pick the best answer and "
            f"explain why, then provide the final answer.\n\n"
            f"Task: {task.description}\n\n"
            f"Final answers:\n{all_answers}"
        )
        jmsg = Message(
            content=judge_prompt,
            role=MessageRole.USER,
            sender=self.id,
            recipient=judge_id,
            metadata={"task_id": task.id, "pattern": "debate_judge"},
        )
        self.memory.add_message(jmsg)
        verdict = judge.process(jmsg)
        self.memory.add_message(verdict)
        return verdict

    # ------------------------------------------------------------------
    # Async wrappers (for use in async applications)
    # ------------------------------------------------------------------
    async def arun_sequential(self, task: Task, agent_ids: list[str]) -> list[Message]:
        return await asyncio.to_thread(self.run_sequential, task, agent_ids)

    async def arun_parallel(self, task: Task, agent_ids: list[str], **kw: Any) -> list[Message]:
        return await asyncio.to_thread(self.run_parallel, task, agent_ids, **kw)

    async def arun_router(self, task: Task, **kw: Any) -> Message:
        return await asyncio.to_thread(self.run_router, task, **kw)

    async def arun_hierarchical(self, task: Task, **kw: Any) -> Message:
        return await asyncio.to_thread(self.run_hierarchical, task, **kw)

    async def arun_debate(self, task: Task, **kw: Any) -> Message:
        return await asyncio.to_thread(self.run_debate, task, **kw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require(self, agent_id: str) -> Agent:
        agent = self.agents.get(agent_id)
        if agent is None:
            raise KeyError(f"No agent registered with id={agent_id!r}")
        return agent

    @staticmethod
    def _process_agent(agent: Agent, message: Message) -> Message:
        return agent.process(message)

    @staticmethod
    def _extract_agent_id(text: str, candidates: list[str]) -> Optional[str]:
        t = text.lower()
        for c in candidates:
            if c.lower() in t:
                return c
        return None

    @staticmethod
    def _parse_plan(text: str, worker_ids: list[str]) -> list[dict[str, str]]:
        """Best-effort parse of a JSON plan from *text*."""
        # Try to find a JSON array in the text.
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                parsed = json.loads(chunk)
                if isinstance(parsed, list):
                    valid = []
                    for item in parsed:
                        if isinstance(item, dict) and "worker" in item and "subtask" in item:
                            if item["worker"] in worker_ids:
                                valid.append({"worker": item["worker"], "subtask": str(item["subtask"])})
                    if valid:
                        return valid
            except json.JSONDecodeError:
                pass
        # Fallback: assign the whole task to the first worker.
        return [{"worker": worker_ids[0], "subtask": text.strip()}] if worker_ids else []

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"Orchestrator '{self.name}' — {len(self.agents)} agents:"]
        for a in self.agents.values():
            lines.append(f"  • {a.id:20s} {a.role}")
        return "\n".join(lines)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)