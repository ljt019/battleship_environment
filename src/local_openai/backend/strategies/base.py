from typing import Protocol, List, Dict, Any

class GuessStrategy(Protocol):
    """Interface that all guessing strategies must follow."""

    def guess(self, messages: List[Dict[str, Any]]): 
        """Return the next coordinate (e.g. "e5")."""
        raise NotImplementedError
