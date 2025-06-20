from typing import List, Dict, Any
from .base import GuessStrategy

class InformationGainStrategy(GuessStrategy):
    """Selects the shot that maximises expected information gain.

    Implementation notes
    --------------------
    • We approximate the posterior over ship locations by enumerating all
      *independent* placements of each still-afloat ship that do not conflict
      with known misses/sunk cells.  This is the same approximation the heat-
      map strategy uses.
    • For each cell we compute

          p = (# of enumerated placements that cover the cell) / (# total placements)

      and take IG(cell) = −[p ln p + (1−p) ln(1−p)] (binary entropy).  This is
      proportional to the expected reduction in log-count entropy.
    • We ignore already-guessed cells and break ties randomly.
    """

    _SHIP_SIZES = {
        "carrier": 5,
        "battleship": 4,
        "cruiser": 3,
        "submarine": 3,
        "destroyer": 2,
    }

    def guess(self, messages: List[Dict[str, Any]]):
        import re, math, random as _rand

        # ----------- extract latest <state> and <remaining> tags ---------- #
        remaining_tag = None
        state_tag = None
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if remaining_tag is None and "<remaining" in content:
                remaining_tag = content
            if state_tag is None and "<state" in content:
                state_tag = content
            if remaining_tag and state_tag:
                break

        # ------------------ parse board info ----------------------------- #
        hits: set[str] = set()
        misses: set[str] = set()
        sunk: set[str] = set()

        if state_tag:
            def _extract(attr: str) -> list[str]:
                m = re.search(rf'{attr}="([^"]*)"', state_tag)
                return m.group(1).strip().split() if m else []

            hits = set(_extract("hits"))
            misses = set(_extract("misses"))
            sunk = set(_extract("sunk"))

        guessed = hits | misses | sunk

        # Remaining ships
        remaining_ships: list[int] = []
        if remaining_tag:
            for name, size in self._SHIP_SIZES.items():
                m = re.search(rf'{name}="(\d+)"', remaining_tag)
                count = int(m.group(1)) if m else 1
                remaining_ships.extend([size] * count)
        else:
            remaining_ships = list(self._SHIP_SIZES.values())

        # ------------ enumerate placements & accumulate counts ----------- #
        heat = [[0 for _ in range(10)] for _ in range(10)]
        total_placements = 0

        def rc_to_coord(r: int, c: int) -> str:
            return f"{chr(ord('a') + c)}{r + 1}"

        def coord_to_rc(coord: str) -> tuple[int, int]:
            col = ord(coord[0].lower()) - ord('a')
            row = int(coord[1:]) - 1
            return row, col

        def placement_ok(cells: list[str]) -> bool:
            # Must not overlap misses/sunk
            if any(cell in misses or cell in sunk for cell in cells):
                return False
            # Optional: ensure consistency with hits (either covers or not?) –
            # we allow placements whether or not they cover hits; this matches
            # the standard heat-map approximation and works well in practice.
            return True

        for ship_len in remaining_ships:
            # Horizontal placements
            for r in range(10):
                for c in range(0, 10 - ship_len + 1):
                    cells = [rc_to_coord(r, c + k) for k in range(ship_len)]
                    if placement_ok(cells):
                        total_placements += 1
                        for cell in cells:
                            rr, cc = coord_to_rc(cell)
                            heat[rr][cc] += 1

            # Vertical placements
            for r in range(0, 10 - ship_len + 1):
                for c in range(10):
                    cells = [rc_to_coord(r + k, c) for k in range(ship_len)]
                    if placement_ok(cells):
                        total_placements += 1
                        for cell in cells:
                            rr, cc = coord_to_rc(cell)
                            heat[rr][cc] += 1

        # ---------------- choose cells to consider ----------------------- #

        # Target-first: if there are unsunk hits, restrict to 4-connected
        # neighbours that are still unknown; otherwise evaluate every
        # unknown cell on the board.

        def _neighbours(coord: str):
            r, c = coord_to_rc(coord)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    yield rc_to_coord(nr, nc)

        unsunk_hits = hits - sunk

        target_candidates = {
            nb for h in unsunk_hits for nb in _neighbours(h) if nb not in guessed
        }

        if target_candidates:
            cells_to_consider = target_candidates
        else:
            # All unknown cells
            cells_to_consider = {
                rc_to_coord(r, c)
                for r in range(10)
                for c in range(10)
                if rc_to_coord(r, c) not in guessed
            }

        # -------------- compute information gain per cell ---------------- #
        best_ig = -1.0
        best_cells: list[str] = []

        for coord in cells_to_consider:
            r, c = coord_to_rc(coord)
            if coord in guessed:
                continue
            count = heat[r][c]
            if count == 0 or count == total_placements:
                ig = 0.0
            else:
                p = count / total_placements
                ig = - (p * math.log(p) + (1 - p) * math.log(1 - p))
            if ig > best_ig + 1e-9:
                best_ig = ig
                best_cells = [coord]
            elif abs(ig - best_ig) < 1e-9:
                best_cells.append(coord)

        if not best_cells:
            # Shouldn't happen, but fallback to random unknown cell.
            best_cells = list(cells_to_consider) if cells_to_consider else [rc_to_coord(r, c) for r in range(10) for c in range(10) if rc_to_coord(r, c) not in guessed]

        return _rand.choice(best_cells)