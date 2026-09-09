# 🎼 Agent Orchestra

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-hobby%2F%23-lightgrey)
![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen)

一个轻量级、**零依赖**的多智能体编排框架，用 Python 实现。

支持多种编排模式：顺序流水线、并行扇出、路由分发、层级委派、辩论、以及声明式 DAG 工作流。

> 开箱即用：内置 `MockBackend` 让你在没有 API Key 的情况下也能完整运行和测试。接入真实 LLM 只需实现一个 `LLMBackend` 接口。

---

## 🤔 Why

已经有 LangGraph、CrewAI、AutoGen 了，为什么再造一个？

- **零依赖**：不依赖任何第三方库（仅测试用 pytest），pip install 即可跑，没有依赖地狱
- **教学/原型导向**：代码精简、模式清晰，适合理解多智能体编排的核心概念，也适合快速验证想法
- **轻量内聚**：核心 6 个模块，每个职责单一，改一个模式不影响其他
- **MockBackend**：离线可跑完整流程，接入真实 LLM 时只换一行代码

**不是什么**：不是 LangGraph 的替代品，不适合生产级高并发场景。

---

## 📋 Prerequisites

- Python **3.10+**
- 无外部运行依赖（`pip install -e .` 仅安装包元数据）

---

## ✨ 特性

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Sequential** | 智能体按顺序执行，上一步输出传入下一步 | 研究→写作→审校流水线 |
| **Parallel** | 多个智能体并发处理同一任务，可聚合 | 多视角头脑风暴 |
| **Router** | 路由智能体分析任务，选择最合适的候选智能体 | 动态任务分发 |
| **Hierarchical** | 管理者拆解任务→委派给工人→综合结果 | 复杂项目分解 |
| **Debate** | 多智能体多轮辩论，裁判选出最佳答案 | 需要权衡的决策 |
| **Workflow DAG** | 声明式有向无环图，支持扇入/扇出 | 复杂依赖编排 |

其他亮点：
- 🧠 **共享记忆（Blackboard）**：所有智能体共享线程安全的上下文与消息历史
- 🔌 **LLM 后端抽象**：内置 `MockBackend` / `ScriptedBackend` / `OpenAIBackend`（OpenAI 兼容，可接 DeepSeek、SiliconFlow、Ollama）
- 🛠️ **工具调用（Tool use）**：智能体可通过 `TOOL_CALL` 协议调用自定义工具，结果自动回填
- 🏷️ **能力声明**：智能体声明 `AgentCapability`，路由器据此匹配
- ⚡ **并发执行**：基于线程池的并行调度 + `asyncio` 异步包装
- 🧪 **完整测试**：49 个单元测试，覆盖所有模式

---

## 📦 安装

```bash
cd agent_orchestra
pip install -e .          # 可编辑安装
# 或仅安装测试依赖：
pip install -e ".[test]"
```

无需安装即可直接运行：

```bash
python agent_orchestra/examples/demo.py
```

安装后可用控制台命令：

```bash
pip install -e ".[openai]"   # 需要真实 LLM 时安装 openai 后端依赖
agent-orchestra-demo
```

---

## 🚀 快速开始

### 顺序流水线

```python
from agent_orchestra import Orchestrator, Task, create_default_team

orch = Orchestrator("my-team")
orch.register_many(*create_default_team())   # 注册 7 个内置智能体

results = orch.run_sequential(
    Task("写一篇关于可再生能源的短文。"),
    agent_ids=["researcher", "writer", "analyst"],
)
print(results[-1].content)   # analyst 的最终输出
```

### 并行 + 聚合

```python
results = orch.run_parallel(
    Task("为智能家居系统头脑风暴方案。"),
    agent_ids=["researcher", "writer", "coder"],
    aggregator="planner",          # planner 综合三方输出
)
```

### 路由分发

```python
result = orch.run_router(
    Task("调试一个抛出 KeyError 的 Python 函数。"),
    router_agent="router",
    candidate_ids=["researcher", "writer", "coder", "analyst"],
)
print(f"路由到: {result.sender}")
```

### 层级委派

```python
final = orch.run_hierarchical(
    Task("制作一份 2025 年 AI 智能体迷你报告，含代码示例。"),
    manager_agent="planner",
    worker_ids=["researcher", "coder", "analyst"],
)
```

### 辩论

```python
verdict = orch.run_debate(
    Task("学编程最好的方式：项目还是教程？"),
    debater_ids=["writer", "coder", "analyst"],
    rounds=2,
    judge_agent="judge",
)
```

### 声明式 DAG 工作流

```python
from agent_orchestra import Workflow, WorkflowEngine

team = {a.id: a for a in create_default_team()}
engine = WorkflowEngine(team)

wf = (
    Workflow("report-pipeline")
    .step("research", "researcher")
    .step("write", "writer", depends_on=["research"])
    .step("code", "coder", depends_on=["research"])     # 与 write 并行
    .step("review", "analyst", depends_on=["write", "code"])  # 汇合
)

results = engine.run(wf, Task("写一份 Python 装饰器指南。"))
print(results["review"].content)
```

---

## 🎬 Demo 输出示例

```
======================================================================
  1. SEQUENTIAL PIPELINE  (researcher -> writer -> analyst)
======================================================================
...
[sequential step 2] analyst -> sequential-team
Analysis:
- Strengths: clarity, structure.
- Weaknesses: needs more evidence.
- Verdict: APPROVE with minor revisions.

======================================================================
  3. ROUTER  (router picks the best agent for each task)
======================================================================
--- Task: Debug a Python function that throws KeyError.
Routed to: coder
--- Task: Write a blog post intro about space exploration.
Routed to: writer
--- Task: Analyze the strengths of this marketing copy.
Routed to: analyst
  DONE — all patterns executed successfully ✅
```

> 完整运行：`python agent_orchestra/examples/demo.py` 或安装后执行 `agent-orchestra-demo`（无需 API Key，30 秒内完成）

---

## 🏗️ 架构

```
agent_orchestra/
├── core/
│   ├── message.py        # Message, Task, MessageRole — 通信协议
│   ├── memory.py         # Memory — 线程安全的共享黑板
│   ├── llm.py            # LLMBackend, MockBackend, ScriptedBackend, OpenAIBackend
│   ├── agent.py          # Agent 基类, AgentCapability, Tool
│   ├── orchestrator.py   # Orchestrator — 5 种编排模式
│   └── workflow.py       # Workflow, Step, WorkflowEngine — DAG 引擎
├── agents/
│   └── builtin.py        # 7 个内置智能体 + 工厂函数
├── examples/
│   └── demo.py           # 全模式演示（安装后可用 agent-orchestra-demo）
└── tests/
    └── test_orchestra.py # 单元测试
```

### 核心概念

- **Message** — 智能体间通信的原子单元，含 `role`/`sender`/`recipient`/`metadata`
- **Task** — 待分派的工作单元，可转换为 Message
- **Agent** — 拥有角色、系统提示词、能力声明和 LLM 后端；实现 `process(message) -> message`，可挂载 `Tool`
- **Tool** — 可调用能力：智能体输出 `TOOL_CALL` 标记，框架执行工具并回填结果
- **Orchestrator** — 注册中心 + 调度器，管理共享记忆，提供 5 种编排模式
- **Workflow** — 声明式 DAG，`WorkflowEngine` 按拓扑序并发执行

### 内置智能体

| 智能体 | id | 角色 |
|--------|----|------|
| ResearcherAgent | `researcher` | 调研、收集信息、总结 |
| WriterAgent | `writer` | 撰写清晰结构化内容 |
| AnalystAgent | `analyst` | 批判性评审、给出裁决 |
| CoderAgent | `coder` | 编写、审查、调试代码 |
| PlannerAgent | `planner` | 任务分解与协调 |
| RouterAgent | `router` | 分析任务并路由 |
| JudgeAgent | `judge` | 比较多个答案选出最佳 |

---

## 📖 API 参考

### 核心类

| 类 | 模块 | 说明 |
|----|------|------|
| `Agent` | `core.agent` | 基类。子类化并重写 `process` / `build_prompt`，或仅设置 `system_prompt`。 |
| `AgentCapability` | `core.agent` | 声明智能体能力（用于路由匹配）。 |
| `Message` | `core.message` | 智能体间通信的原子单元。 |
| `Task` | `core.message` | 待分派的工作单元，可转换为 `Message`。 |
| `MessageRole` | `core.message` | 消息角色枚举：`SYSTEM`、`USER`、`ASSISTANT`、`TOOL`。 |
| `Memory` | `core.memory` | 线程安全的共享黑板（键值存储 + 消息历史）。 |
| `LLMBackend` | `core.llm` | 所有 LLM 后端必须实现的抽象接口。 |
| `MockBackend` | `core.llm` | 确定性、基于关键词的后端，用于离线演示和测试。 |
| `ScriptedBackend` | `core.llm` | 按顺序返回预定义响应，用于单元测试。 |
| `OpenAIBackend` | `core.llm` | OpenAI 兼容后端，可接 OpenAI / DeepSeek / SiliconFlow / Ollama。 |
| `Tool` | `core.agent` | 可调用能力：智能体输出 `TOOL_CALL` 标记即触发执行。 |
| `Orchestrator` | `core.orchestrator` | 注册中心 + 调度器，提供 5 种编排模式。 |
| `Workflow` | `core.workflow` | 声明式 DAG，通过 `WorkflowEngine` 执行。 |
| `WorkflowEngine` | `core.workflow` | 按拓扑序并发执行 `Workflow`。 |
| `Step` | `core.workflow` | DAG 中的单个节点。 |

### Orchestrator 方法

| 方法 | 说明 |
|------|------|
| `register(agent)` | 注册智能体并绑定到共享记忆。 |
| `register_many(*agents)` | 批量注册智能体。 |
| `send(message)` | 将消息路由到目标智能体并返回响应。 |
| `dispatch(task)` | 将任务分派给 `task.assigned_to` 指定的智能体。 |
| `run_sequential(task, agent_ids, pass_output=True)` | 顺序流水线，每步输出传入下一步。 |
| `run_parallel(task, agent_ids, aggregator=None)` | 并行扇出，可选聚合智能体。 |
| `run_router(task, router_agent, candidate_ids)` | 路由智能体选择最佳候选者。 |
| `run_hierarchical(task, manager_agent, worker_ids)` | 管理者分解任务→委派→综合。 |
| `run_debate(task, debater_ids, rounds, judge_agent)` | 多轮辩论，裁判选出最佳答案。 |
| `arun_sequential(task, agent_ids)` | `run_sequential` 的异步包装。 |
| `arun_parallel(task, agent_ids, **kw)` | `run_parallel` 的异步包装。 |
| `arun_router(task, **kw)` | `run_router` 的异步包装。 |
| `arun_hierarchical(task, **kw)` | `run_hierarchical` 的异步包装。 |
| `arun_debate(task, **kw)` | `run_debate` 的异步包装。 |
| `summary()` | 返回编排器状态的文本摘要。 |
| `shutdown()` | 关闭线程池。 |

### Workflow 方法

| 方法 | 说明 |
|------|------|
| `step(id, agent_id, depends_on=None, prompt_template=…, description="")` | 流畅构建器，添加一个步骤。 |
| `add_step(step)` | 添加一个 `Step` 实例。 |
| `roots()` | 返回无上游依赖的步骤 ID 列表。 |
| `validate()` | 检查图是否为有效 DAG（无环、依赖存在）。 |
| `WorkflowEngine.run(workflow, task)` | 执行工作流，返回 `{step_id: Message}` 映射。 |

### LLMBackend 接口

```python
class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt, *, system_prompt="", **kwargs) -> str: ...

    def __call__(self, prompt, **kwargs) -> str: ...
```

### Memory API

| 方法 | 说明 |
|------|------|
| `set(key, value)` | 存储键值对。 |
| `get(key, default=None)` | 获取值，不存在时返回 `default`。 |
| `append(key, value)` | 追加到列表值（不存在则创建）。 |
| `delete(key)` | 删除键。 |
| `add_message(message)` | 添加消息到历史记录。 |
| `history` | 完整消息历史（只读副本）。 |
| `recent(n=5)` | 返回最近 *n* 条消息。 |
| `snapshot()` | 返回键值存储的浅拷贝。 |
| `clear()` | 清空存储和历史。 |
| `__contains__(key)` | `key in memory`。 |
| `__getitem__(key)` | `memory[key]` 等价于 `get(key)`。 |
| `__setitem__(key, value)` | `memory[key] = value` 等价于 `set(key, value)`。 |

---

## 🔌 接入真实 LLM

内置 OpenAI 兼容后端 `OpenAIBackend`，只需换后端即可接入真实模型：

```python
from agent_orchestra import Orchestrator, OpenAIBackend, create_default_team

backend = OpenAIBackend(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",   # OpenAI / DeepSeek / SiliconFlow / Ollama
    model="deepseek-chat",
)

orch = Orchestrator(llm=backend)
orch.register_many(*create_default_team(llm=backend))
```

### 使用工具（Tool use）

给智能体挂载自定义工具，通过 `TOOL_CALL` 协议调用：

```python
from agent_orchestra import Agent, Tool

def get_weather(city: str) -> str:
    return f"{city}: 24°C, sunny"

agent = Agent(
    name="Assistant",
    role="assistant",
    llm=backend,
    tools=[Tool(name="get_weather", description="查询天气", function=get_weather)],
)
```

智能体在回答中输出 `TOOL_CALL: get_weather({"city": "Shanghai"})`，框架会自动执行并把结果回填到下一轮 LLM 上下文。

实现自定义后端时，实现 `LLMBackend` 接口即可（`generate(prompt, *, system_prompt, **kwargs) -> str`）。

---

## 🧪 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 或不安装 pytest 直接运行
python tests/test_orchestra.py
```

---

## 📅 Roadmap

> Coming soon. 这个项目目前处于 v0.1.0 的早期阶段，后续计划将在这里更新。

---

## 📄 许可证

MIT
