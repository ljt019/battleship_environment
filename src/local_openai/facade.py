"""
Tiny facade that mimics openai.OpenAI's surface but delegates to a pluggable
Backend object.  Only .completions.create, .chat.completions.create and base_url
are implemented because that's all verifiers needs.
"""
from types import SimpleNamespace
from typing import Any, Dict, List

class BackendProtocol:
    def completion(
        self,
        *,
        model: str,
        is_chat: bool,
        messages: List[Dict[str, str]] | None = None,
        prompt: str | None = None,
        **sampling: Any,
    ) -> Any: ...

class LocalOpenAI:
    """Drop-in stand-in for openai.OpenAI that stays entirely in-process."""

    class _CompletionsEndpoint:
        def __init__(self, backend: BackendProtocol):
            self._backend = backend

        def create(self, *, model: str, prompt: str, **kw):
            return self._backend.completion(
                model=model, prompt=prompt, is_chat=False, **kw
            )

    class _ChatCompletionsEndpoint:
        def __init__(self, backend: BackendProtocol):
            self._backend = backend

        def create(self, *, model: str, messages: List[Dict[str, str]], **kw):
            return self._backend.completion(
                model=model, messages=messages, is_chat=True, **kw
            )

    class _ChatEndpoint:
        def __init__(self, backend: BackendProtocol):
            self.completions = LocalOpenAI._ChatCompletionsEndpoint(backend)

    def __init__(self, backend: BackendProtocol):
        self._backend = backend
        self.completions = LocalOpenAI._CompletionsEndpoint(backend)
        self.chat = LocalOpenAI._ChatEndpoint(backend)

        # dummy base_url
        self.base_url = "http://localhost"
