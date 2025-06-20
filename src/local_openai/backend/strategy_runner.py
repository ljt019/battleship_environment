from types import SimpleNamespace
from typing import Any, Dict, List
from .strategies.base import GuessStrategy
from .strategies.random_guess import RandomGuessStrategy

class StrategyRunnerBackend:
    def __init__(self, strategy: GuessStrategy | None = None):
        self._strategy = strategy or RandomGuessStrategy()

    def completion(
        self,
        *,
        model: str = "battleship-dummy-model",
        is_chat: bool,
        messages: List[Dict[str, str]] | None = None,
        prompt: str | None = None,
        **_: Any,
    ):
        coord = self._strategy.guess(messages or [])
        content = f"<think></think>\n\n<guess>[{coord}]</guess>"

        if is_chat:
            msg = SimpleNamespace(role="assistant", content=content)
            choice = SimpleNamespace(message=msg, finish_reason="stop")
        else:
            choice = SimpleNamespace(text=content, finish_reason="stop")

        return SimpleNamespace(choices=[choice])
