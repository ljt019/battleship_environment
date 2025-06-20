from typing import List, Dict, Any
from .base import GuessStrategy

class ParityHuntTargetStrategy(GuessStrategy):
    """Checkerboard hunt that switches to *target* mode after a hit.

    1. Parse the most recent environment `<state>` tag to obtain sets of
       hits, misses and sunk cells.
    2. If there is at least one *unsunk* hit, enter *target* mode:
       - queue the 4-connected neighbours (up/down/left/right) of every such
         hit that are still unknown.
       - return the first candidate (ties broken randomly).
    3. Otherwise (*hunt* mode) fire on the 50 % parity mask where
       `(row + col) % 2 == 0`, excluding already guessed cells.

    This guarantees never wasting a shot on a square that cannot contain the
    2-length ship.
    """

    _ALL_COORDS = [
        f"{chr(ord('a') + c)}{r+1}"
        for r in range(10)
        for c in range(10)
    ]

    # ----------------------------- helpers ----------------------------- #
    @staticmethod
    def _coord_to_rc(coord: str) -> tuple[int, int]:
        col = ord(coord[0].lower()) - ord('a')
        row = int(coord[1:]) - 1
        return row, col

    @staticmethod
    def _rc_to_coord(row: int, col: int) -> str:
        return f"{chr(ord('a') + col)}{row + 1}"

    @classmethod
    def _get_neighbours(cls, coord: str) -> list[str]:
        row, col = cls._coord_to_rc(coord)
        candidates = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                candidates.append(cls._rc_to_coord(nr, nc))
        return candidates

    # ------------------------------ main ------------------------------ #
    def guess(self, messages: List[Dict[str, Any]]):
        import random, re

        # 1) Locate the last environment message containing a <state> tag.
        state_tag = None
        for msg in reversed(messages):
            if msg.get("role") == "user" and "<state" in msg.get("content", ""):
                state_tag = msg["content"]
                break

        hits: set[str] = set()
        misses: set[str] = set()
        sunk: set[str] = set()

        if state_tag:
            # Parse attribute lists using regex.
            def _extract(attr: str) -> list[str]:
                m = re.search(rf'{attr}="([^"]*)"', state_tag)
                return m.group(1).strip().split() if m else []

            hits = set(_extract("hits"))
            misses = set(_extract("misses"))
            sunk = set(_extract("sunk"))

        guessed = hits | misses | sunk

        # ------------------------------------------------------------------
        # 1. Target phase: if there are unsunk hits, restrict search space
        #    to their 4-connected neighbours that are still unknown.
        # ------------------------------------------------------------------

        unsunk_hits = hits - sunk

        # Helper to collect in-bounds neighbours
        def neighbours(coord: str):
            # Use the class helper methods for coordinate conversions
            r, c = self._coord_to_rc(coord)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    yield self._rc_to_coord(nr, nc)

        target_candidates = {
            nb for h in unsunk_hits for nb in neighbours(h) if nb not in guessed
        }

        # ---------------- Hunt (parity) ----------------
        parity_cells = [c for c in self._ALL_COORDS if ((self._coord_to_rc(c)[0] + self._coord_to_rc(c)[1]) % 2 == 0)]
        hunt_candidates = [c for c in parity_cells if c not in guessed]

        # If parity mask exhausted (rare end-game), fall back to any unknown.
        if not target_candidates:
            target_candidates = set(hunt_candidates)

        # Decide which cells we will evaluate IG for
        cells_to_consider = target_candidates if target_candidates else {
            self._rc_to_coord(r, c)
            for r in range(10)
            for c in range(10)
            if self._rc_to_coord(r, c) not in guessed
        }

        return random.choice(list(cells_to_consider))