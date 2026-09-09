"""LLM backend abstraction.

The framework talks to LLMs through the :class:`LLMBackend` interface so that
agents can be tested with a deterministic :class:`MockBackend` and later wired
to a real provider (OpenAI, Anthropic, local model, …) without changing agent
code.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMBackend(ABC):
    """Abstract interface every LLM backend must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Return a completion string for *prompt* given *system_prompt*."""

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)


class MockBackend(LLMBackend):
    """Deterministic, keyword-driven backend for offline demos/tests.

    It inspects the prompt + system prompt for keywords and produces
    role-appropriate, templated output.  This keeps the framework fully
    runnable without any API key while still producing interesting,
    differentiated agent behaviour.

    Pass ``responses`` to fully script outputs (popped FIFO), or
    ``rules`` (list of ``(keyword, template)``) for keyword matching.
    """

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        rules: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        """Initialise the MockBackend.

        Args:
            responses: A list of canned responses returned in order.
                When exhausted, the backend falls back to role-aware
                keyword matching.
            rules: A list of ``(keyword, template)`` pairs.  When
                *keyword* is found in the prompt or system prompt,
                the corresponding *template* is used.  Templates
                support ``{prompt}`` and ``{system}`` placeholders.
        """
        self._responses = list(responses) if responses else []
        self._rules = rules or []

    def generate(self, prompt: str, *, system_prompt: str = "", **kwargs: Any) -> str:
        # 1. Fully scripted responses take priority.
        if self._responses:
            return self._responses.pop(0)

        text = f"{system_prompt}\n{prompt}".lower()

        # 2. Keyword rules.
        for keyword, template in self._rules:
            if keyword.lower() in text:
                return template.format(prompt=prompt, system=system_prompt)

        # 3. Role-aware fallback derived from the system prompt.
        return self._role_aware_fallback(prompt, system_prompt)

    # ------------------------------------------------------------------
    @staticmethod
    def _role_aware_fallback(prompt: str, system_prompt: str) -> str:
        sp = system_prompt.lower()
        snippet = prompt.strip().splitlines()[0][:120] if prompt.strip() else ""

        # Order matters: check the most specific role markers first so that
        # e.g. a coder whose prompt contains the word "write" is still
        # recognised as a coder, not a writer.
        if "engineer" in sp or "program" in sp:
            return (
                "```python\n"
                "def solution(data):\n"
                "    # Generated implementation\n"
                f"    # task: {snippet}\n"
                "    return {'status': 'ok', 'result': data}\n"
                "```"
            )
        if "research" in sp or "investigat" in sp:
            return (
                "Research findings:\n"
                f"- Key topic identified: {snippet}\n"
                "- Relevant background gathered from 3 sources.\n"
                "- Open questions noted for follow-up.\n"
                "Recommendation: proceed with synthesis."
            )
        if "content writer" in sp or "expert writer" in sp or "author" in sp:
            return (
                "Draft output:\n"
                f">>> {snippet}\n\n"
                "Here is a polished, structured piece of writing that "
                "addresses the request with a clear introduction, body, "
                "and conclusion."
            )
        if "critical analyst" in sp or "analyst" in sp or "critique" in sp:
            return (
                "Analysis:\n"
                f"- Input: {snippet}\n"
                "- Strengths: clarity, structure.\n"
                "- Weaknesses: needs more evidence.\n"
                "- Verdict: APPROVE with minor revisions."
            )
        if "planner" in sp or "manager" in sp or "coordinat" in sp:
            return (
                "Plan:\n"
                "1. Decompose the task into subtasks.\n"
                "2. Assign each subtask to the best-fit agent.\n"
                "3. Aggregate results and verify quality.\n"
                f"Context: {snippet}"
            )
        if "router" in sp:
            return "researcher"
        if "judge" in sp or "adjudicator" in sp:
            return (
                "Verdict:\n"
                f"- Task: {snippet}\n"
                "- Best answer selected based on accuracy, completeness, "
                "and clarity.\n"
                "- Winner: the response that provided the most actionable "
                "and well-reasoned advice."
            )
        # Generic fallback.
        return f"[MockLLM] Processed request: {snippet}"


class ScriptedBackend(LLMBackend):
    """Backend that returns pre-defined responses in order, then repeats.

    Useful for deterministic unit tests.
    """

    def __init__(self, responses: list[str]) -> None:
        """Initialise the ScriptedBackend.

        Args:
            responses: A list of responses returned in order.
                When the list is exhausted, responses cycle from the
                beginning.
        """
        self._responses = list(responses)
        self._idx = 0

    def generate(self, prompt: str, *, system_prompt: str = "", **kwargs: Any) -> str:
        if not self._responses:
            return ""
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat-completions backend.

    Works with OpenAI, DeepSeek, SiliconFlow, Ollama (OpenAI mode), and any
    other provider exposing the ``/chat/completions`` API: point ``base_url``
    at the provider and pass a valid ``api_key`` (some providers accept a
    placeholder like ``"sk-xxx"`` or ``"ollama"``).

    Requires the optional ``openai`` package::

        pip install "agent-orchestra[openai]"
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        **client_kwargs: Any,
    ) -> None:
        """Initialise the backend.

        Args:
            api_key: API key for the provider.
            base_url: Provider base URL (``…/v1``).
            model: Model name to use for completions.
            timeout: Request timeout in seconds.
            **client_kwargs: Extra kwargs forwarded to the OpenAI client
                (e.g. ``max_retries``).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._timeout = timeout
        self._client_kwargs = client_kwargs
        self._client = None

    def _get_client(self) -> Any:
        """Lazily build the OpenAI client (imported on first use)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "OpenAIBackend requires the 'openai' package. "
                    "Install it with: pip install 'agent-orchestra[openai]'"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self._timeout,
                **self._client_kwargs,
            )
        return self._client

    def generate(self, prompt: str, *, system_prompt: str = "", **kwargs: Any) -> str:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        allowed = {"temperature", "max_tokens", "top_p", "stop", "stream"}
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        params.update({k: v for k, v in kwargs.items() if k in allowed})

        resp = client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""