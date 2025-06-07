import random
import string

class BattleshipGame:
    def __init__(self, board_size=10, ship_sizes=None):
        self.board_size = board_size
        self.cols = string.ascii_lowercase[:board_size]
        self.rows = [str(r) for r in range(1, board_size + 1)]
        self.ship_sizes = ship_sizes if ship_sizes else [5, 4, 3, 3, 2]  # carrier, battleship, cruiser, sub, destroyer

        self.reset()

    def reset(self):
        self.board = {f"{c}{r}": "?" for c in self.cols for r in self.rows}
        self.ships = []
        self.history = []
        self.turn_count = 0
        self.game_over = False

        for size in self.ship_sizes:
            placed = False
            while not placed:
                horiz = random.choice([True, False])
                if horiz:
                    row = random.choice(self.rows)
                    start_idx = random.randint(0, self.board_size - size)
                    coords = [f"{self.cols[start_idx + i]}{row}" for i in range(size)]
                else:
                    col = random.choice(self.cols)
                    start_idx = random.randint(0, self.board_size - size)
                    coords = [f"{col}{self.rows[start_idx + i]}" for i in range(size)]

                # check for overlap
                if not any(c in sum([s["coords"] for s in self.ships], []) for c in coords):
                    self.ships.append({"coords": coords, "hits": set()})
                    placed = True

    def step(self, move):
        """
        move: string like 'b6'
        returns: observation (rendered string), hit (bool), sunk (bool), game_over (bool), invalid_move (bool)
        """
        move = move.lower().strip()
        if move not in self.board or self.board[move] != "?":
            # invalid move
            hit = False
            sunk = False
            self.history.append(move)
            self.turn_count += 1
            return self.render(), hit, sunk, self.game_over, True

        hit = False
        sunk = False

        for ship in self.ships:
            if move in ship["coords"]:
                ship["hits"].add(move)
                hit = True
                if set(ship["hits"]) == set(ship["coords"]):
                    sunk = True
                break

        if hit:
            if sunk:
                self.board[move] = "s"
            else:
                self.board[move] = "x"
        else:
            self.board[move] = "o"

        self.history.append(move)
        self.turn_count += 1

        # check for game over
        if all(set(ship["hits"]) == set(ship["coords"]) for ship in self.ships):
            self.game_over = True

        return self.render(), hit, sunk, self.game_over, False

    def render(self):
        """
        Returns the board + ships remaining + turn history as a text string for LLM prompt.
        """
        lines = []
        header = "  " + " ".join(self.cols)
        lines.append(header)
        for r in self.rows:
            row_cells = []
            for c in self.cols:
                row_cells.append(f"[{self.board[f'{c}{r}']}]")
            lines.append(f"{r.rjust(2)} {' '.join(row_cells)}")

        remaining = []
        for ship in self.ships:
            if set(ship["hits"]) != set(ship["coords"]):
                remaining.append(f"Ship ({len(ship['coords'])})")
        if not remaining:
            remaining.append("None")

        lines.append("")
        lines.append("Remaining ships: " + ", ".join(remaining))
        lines.append("")
        lines.append("Turn history: " + " ".join(self.history) if self.history else "Turn history: (none)")
        lines.append("")
        lines.append("Make a turn like: [b7]")

        return "\n".join(lines)

    def get_valid_moves(self):
        """
        Returns list of available (unknown) cells.
        """
        return [pos for pos, val in self.board.items() if val == "?"]
