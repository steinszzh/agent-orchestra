# Changelog

All notable changes to Agent Orchestra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-09-09

### Added
- `OpenAIBackend` — OpenAI 兼容后端，可接 OpenAI / DeepSeek / SiliconFlow / Ollama。
- Tool use：`Tool` 类 + `TOOL_CALL` 协议，智能体可调用自定义工具并自动回填结果（带 `max_tool_rounds` 防循环上限）。
- 新增 12 个单元测试（工具调用、能力路由回归、OpenAI 后端），总计 49 个。

### Fixed
- `Agent.can_handle` 恒返回 `True` 的 `or True` 缺陷，能力匹配与路由器 fallback 现在真正生效。
- console script `agent-orchestra-demo` 指向包外模块导致安装后无法运行；`examples/` 已移入包内。
- `pyproject.toml` 占位 `Homepage` / `authors` 已更正为真实仓库与作者。
- 删除未使用的 `_hash_seed` 死代码。
- demo 中线程池未释放，补 `shutdown()` 调用。

## [0.1.0] — 2026-08-03

### Added
- Initial release of Agent Orchestra.
- Six orchestration patterns: Sequential, Parallel, Router, Hierarchical, Debate, Workflow DAG.
- Seven built-in agents: Researcher, Writer, Analyst, Coder, Planner, Router, Judge.
- `MockBackend` for offline testing without an API key.
- `ScriptedBackend` for deterministic unit tests.
- `LLMBackend` abstract interface for plugging in real providers (OpenAI, Anthropic, …).
- Thread-safe shared memory (Blackboard) for inter-agent context.
- Capability-based routing via `AgentCapability`.
- 30+ unit tests covering all patterns and core primitives.