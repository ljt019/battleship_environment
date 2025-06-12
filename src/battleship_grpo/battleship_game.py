import random
import string

class BattleshipGame:
    def __init__(self, board_size=10, ship_sizes=None):
        self.board_size = board_size
        self.cols = string.ascii_lowercase[:board_size]
        self.rows = [str(r) for r in range(1, board_size + 1)]
        self.ship_sizes = ship_sizes if ship_sizes else [5, 4, 3, 3, 2]  # carrier, battleship, cruiser, sub, destroyer
        self.ship_names = {5: "Carrier", 4: "Battleship", 3: "Cruiser/Submarine", 2: "Destroyer"}

        self.reset()

    def reset(self):
        self.board = {f"{c}{r}": "?" for c in self.cols for r in self.rows}
        self.ships = []
        self.history = []
        self.turn_count = 0
        self.game_over = False
        self.last_move_invalid = False

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
            self.last_move_invalid = True
            self.history.append(move)
            self.turn_count += 1
            return self.render(), hit, sunk, self.game_over, True

        hit = False
        sunk = False
        sunk_ship = None
        self.last_move_invalid = False

        for ship in self.ships:
            if move in ship["coords"]:
                ship["hits"].add(move)
                hit = True
                if set(ship["hits"]) == set(ship["coords"]):
                    sunk = True
                    sunk_ship = ship
                break

        if hit:
            if sunk:
                # Mark all positions of the sunk ship as 's'
                for coord in sunk_ship["coords"]:
                    self.board[coord] = "s"
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
        header = "    " + "".join(f"{c}  " for c in self.cols)
        lines.append(header)
        for r in self.rows:
            row_cells = []
            for c in self.cols:
                row_cells.append(f"[{self.board[f'{c}{r}']}]")
            lines.append(f"{r.rjust(2)} {''.join(row_cells)}")

        remaining = []
        ship_counts = {}
        for ship in self.ships:
            if set(ship["hits"]) != set(ship["coords"]):
                size = len(ship["coords"])
                if size == 3:
                    # Handle both cruiser and submarine (both size 3)
                    ship_counts[3] = ship_counts.get(3, 0) + 1
                    count = ship_counts[3]
                    if count == 1:
                        name = "Cruiser"
                    else:
                        name = "Submarine"
                else:
                    name = self.ship_names[size]
                remaining.append(f"{name} ({size})")
        if not remaining:
            remaining.append("None")

        lines.append("")
        lines.append("Remaining ships: " + ", ".join(remaining))
        if self.last_move_invalid:
            lines.append("")
            lines.append("WARNING: Previous move was INVALID and wasted a turn!")
        lines.append("")

        return "\n".join(lines)

    def get_valid_moves(self):
        """
        Returns list of available (unknown) cells.
        """
        return [pos for pos, val in self.board.items() if val == "?"]

    def render_compact(self):
        """Coordinate=value listing, e.g. a1=. b1=. … j10=o"""
        lines = []
        for r in self.rows:
            cells = []
            for c in self.cols:
                val = self.board[f"{c}{r}"]
                cells.append(f"{c}{r}={val}")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def compact_to_pretty(compact_str):
        """Convert the coordinate=value compact board (from `render_compact`) into a
        bracketed grid similar to `render()` but using a dot (.) for unknown squares.

        Example compact line:  "a1=? b1=o c1=x … j1=?"
        Result pretty row:     " 1 [.][o][x]…[.]"
        """

        if not compact_str.strip():
            return compact_str  # Nothing to convert

        # Build a dictionary of cell -> value
        cell_map = {}
        for token in compact_str.replace("\n", " ").split():
            if "=" not in token:
                continue
            coord, val = token.split("=", 1)
            cell_map[coord.lower()] = val

        if not cell_map:
            return compact_str  # Unexpected format; bail out

        # Determine columns and rows present
        cols = sorted({c[0] for c in cell_map.keys()})
        rows = sorted({int(c[1:]) for c in cell_map.keys()})

        # Compute width of row labels (e.g. 1–10)
        row_label_width = len(str(max(rows)))
        # Header indent: row-label-width + 1 space so the first column letter aligns over the first cell
        header = " " * (row_label_width) + " ".join(cols)
        pretty_lines = [header]

        for r in rows:
            row_chars = []
            for col in cols:
                char = cell_map.get(f"{col}{r}", "?")
                display = "." if char == "?" else char
                row_chars.append(display)
            # Row label aligned to the same width, followed by a single space
            pretty_lines.append(f"{str(r).rjust(row_label_width)} {' '.join(row_chars)}")

        return "\n".join(pretty_lines)
