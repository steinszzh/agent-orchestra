# Changelog

All notable changes to Agent Orchestra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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