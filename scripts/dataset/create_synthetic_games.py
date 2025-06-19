import random

from openai import OpenAI
from datasets import Dataset

from src.battleship_grpo import BattleshipEnv
from typing import Protocol, List, Dict, Any
from scripts.config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL

TOP_PERCENTAGE = 1
NUM_SAMPLES = 1
MAX_TURNS = 10

def main():
    fake_client = FakeOpenAIClient(guess_strategy=InformationGainStrategy())

    vf_env = BattleshipEnv(
        seed=random.randint(0, 1000000),
        max_concurrent=32,
        max_turns=MAX_TURNS
    )

    # Generate synthetic games
    results = vf_env.evaluate(
        client=fake_client,
        model="dummy-model-name",
        num_samples=NUM_SAMPLES,
    )

    dataset = vf_env.make_dataset(results)
    dataset = dataset.sort("reward", reverse=True).select(range(int(len(dataset) * TOP_PERCENTAGE)))

    # Convert multi-turn episodes into single-turn training samples
    turn_rows = []
    for game in dataset:
        full_prompt = game["prompt"]
        full_completion = game["completion"] 
        msgs = full_prompt + full_completion
        episode_reward = game["reward"]

        static_msgs = full_prompt[:2]

        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue

            board_msg = None
            for j in range(i - 1, -1, -1):
                if msgs[j].get("role") == "user" and "<grid>" in msgs[j].get("content", ""):
                    board_msg = msgs[j]
                    break

            if board_msg is None:
                continue

            prompt_turn = static_msgs + [board_msg]
            completion_turn = [m]

            turn_rows.append({
                "prompt": prompt_turn,
                "completion": completion_turn,
                "reward": episode_reward,
                "answer": game.get("answer", ""),
                "task": game.get("task", None),
            })

    dataset = Dataset.from_list(turn_rows)

    openrouter_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    dataset_rows = []
    for turn in dataset: 
        # get the prompt and completion
        turn_prompt = turn["prompt"]
        turn_completion = turn["completion"]

        game_prompt = turn_prompt + turn_completion
        system_prompt = "You are filling in the missing <think> traces for the following move in the game of battleship. Generate your though process as if you were deciding how to maek the move, not as if it was already decided. Only respond with the content of the <think> tag, everything in your response will be placed between the tags."

        openrouter_response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": game_prompt}
            ]
        )

        # splice the response inbetween the <think> and </think> tags in the turn_completion
        turn_completion[0]["content"] = turn_completion[0]["content"].replace("<think>", openrouter_response.choices[0].message.content)
        turn_completion[0]["content"] = turn_completion[0]["content"].replace("</think>", "")
        turn["completion"] = turn_completion

        dataset_rows.append(turn)
    
    dataset = Dataset.from_list(dataset_rows)
    dataset.push_to_hub("battleship-synthetic-games-dataset")

# ---------------------------------------------------------------------------
# Guessing strategy abstraction
# ---------------------------------------------------------------------------

class GuessStrategy(Protocol):
    """Interface that all guessing strategies must follow."""

    def guess(self, messages: List[Dict[str, Any]]): 
        """Return the next coordinate (e.g. "e5")."""
        raise NotImplementedError

class RandomGuessStrategy:
    """Uniform random coordinate picker – default placeholder."""

    def guess(self, messages): 
        import random, string

        col = random.choice(string.ascii_lowercase[:10])
        row = random.randint(1, 10)
        return f"{col}{row}"

class ParityHuntTargetStrategy:
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

class FakeOpenAIClient:
    """Minimal fake replacement for the OpenAI Python client.

    This class implements just enough of the interface expected by
    `verifiers.MultiTurnEnv` (via `BattleshipEnv`) to be usable when you
    want to generate **synthetic** games without calling a real LLM.  It
    always returns an empty `<think>` block and a random Battleship guess
    chosen uniformly from the 10×10 grid (``a1``–``j10``).

    Only the ``chat.completions.create`` endpoint is needed for this use
    case, but a plain ``completions.create`` stub is provided as well so
    that the object behaves like the genuine OpenAI client.
    """

    # ------------------------------------------------------------------
    # Public API (mimicking ``openai.OpenAI``)
    # ------------------------------------------------------------------
    def __init__(self, guess_strategy: "GuessStrategy | None" = None):
        """Create the fake client.

        Parameters
        ----------
        guess_strategy : GuessStrategy | None, optional
            A callable / object that decides the next Battleship coordinate
            when the environment asks for a move.  If *None* (default) a
            uniform random picker is used.  Implementing your own strategy is
            as simple as creating a class with a

            ``def guess(self, messages: list[dict[str, str]]) -> str``

            method and passing an instance here.
        """

        # Needed for `Environment.sanitize_sampling_args`
        self.base_url = "http://localhost/v1"

        # Plug-and-play guesser
        if guess_strategy is None:
            guess_strategy = RandomGuessStrategy()
        self._guess_strategy: GuessStrategy = guess_strategy

        # Sub-endpoints (shape matches the OpenAI client)
        self.completions = self._CompletionsEndpoint(self)
        self.chat = self._ChatEndpoint(self)

    # ----------------------- strategy plumbing ------------------------
    def set_guess_strategy(self, strategy: "GuessStrategy") -> None: 
        """Replace the current guessing strategy at runtime."""
        self._guess_strategy = strategy

    # Convenience wrapper used by the endpoint classes
    def _generate_guess(self, messages): 
        return self._guess_strategy.guess(messages)

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _random_coordinate() -> str:
        """Return a random Battleship coordinate (e.g. "e5")."""
        import random, string

        col = random.choice(string.ascii_lowercase[:10])
        row = random.randint(1, 10)
        return f"{col}{row}"

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------
    class _CompletionsEndpoint:
        """Imitates the ``.completions.create`` endpoint but is unused here."""
        def __init__(self, parent: "FakeOpenAIClient"):
            self._parent = parent

        def create(self, model: str, prompt: str, **kwargs): 
            from types import SimpleNamespace

            coord = self._parent._generate_guess(messages=None)
            text = f"<think></think>\n\n<guess>[{coord}]</guess>"
            choice = SimpleNamespace(text=text, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

    # -------------------------- CHAT ----------------------------------
    class _ChatEndpoint:
        def __init__(self, parent: "FakeOpenAIClient"):
            self.completions = FakeOpenAIClient._ChatCompletionsEndpoint(parent)

    class _ChatCompletionsEndpoint:
        """Mimics ``.chat.completions.create`` used by the environment."""
        def __init__(self, parent: "FakeOpenAIClient"):
            self._parent = parent

        def create(self, model: str, messages, **kwargs): 
            """Return a dummy assistant reply containing a random guess."""
            from types import SimpleNamespace

            coord = self._parent._generate_guess(messages)
            content = f"<think></think>\n\n<guess>[{coord}]</guess>"
            msg = SimpleNamespace(role="assistant", content=content)
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

# ---------------------------------------------------------------------------
# Information-Gain (Entropy) strategy
# ---------------------------------------------------------------------------

class InformationGainStrategy:
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

if __name__ == "__main__":
    main() 