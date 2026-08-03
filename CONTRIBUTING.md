# Contributing to Agent Orchestra

Thank you for your interest in contributing! This guide covers how to get
started with development, what conventions to follow, and how to submit
changes.

## Getting Started

1. Fork the repository and clone your fork locally.
2. Install the project in editable mode with test dependencies:

   ```bash
   pip install -e ".[test]"
   ```

3. Run the test suite to verify everything passes:

   ```bash
   python -m pytest tests/ -v
   ```

## Project Structure

```
agent_orchestra/
├── core/           # Framework primitives (Agent, Orchestrator, Memory, …)
├── agents/         # Built-in agent implementations
├── examples/       # Demo scripts
├── tests/          # Unit tests
└── README.md       # Project overview and quick start
```

## Conventions

- **Docstrings**: All public classes and methods must have docstrings.
  Use triple-quoted strings with Args/Returns sections where applicable.
- **Type hints**: All function signatures must include type annotations.
- **Testing**: New features must include unit tests in `tests/test_orchestra.py`.
- **Zero dependencies**: The core framework must remain dependency-free.
  Optional dependencies (e.g., `openai`, `anthropic`) are declared in
  `pyproject.toml` under `[project.optional-dependencies]`.

## Adding a New Agent

1. Subclass `Agent` in `agent_orchestra/agents/builtin.py`.
2. Provide a `name`, `role`, `description`, `system_prompt`, and
   `capabilities`.
3. Add the new agent to `create_default_team()`.
4. Export it from `agent_orchestra/agents/__init__.py`.
5. Add it to the `__all__` list in `agent_orchestra/__init__.py`.
6. Write a test for it in `tests/test_orchestra.py`.
7. Update the README built-in agents table.

## Adding a New Orchestration Pattern

1. Add the method to `Orchestrator` in `core/orchestrator.py`.
2. Add an async wrapper (`arun_*`) if applicable.
3. Add the pattern to the README feature table and quick-start section.
4. Write tests covering the new pattern.
5. Add a demo case in `examples/demo.py`.

## Submitting Changes

1. Ensure all tests pass: `python -m pytest tests/ -v`.
2. Commit with a clear, concise message.
3. Open a pull request describing the change and its motivation.