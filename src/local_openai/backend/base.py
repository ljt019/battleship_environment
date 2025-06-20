from typing import Protocol, List, Dict, Any

class Backend(Protocol):
    def completion(
        self,
        *,
        model: str,
        is_chat: bool,
        messages: List[Dict[str, str]] | None = None,
        prompt: str | None = None,
        **sampling: Any,
    ): ...
