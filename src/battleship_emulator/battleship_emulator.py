import random
import string

from .models import BoardState, GameState, ShipStatus


class BattleshipEmulator:
    def __init__(self, board_size=10, ship_sizes=None, seed=None):
        self.board_size = board_size
        self.cols = string.ascii_lowercase[:board_size]
        self.rows = [str(r) for r in range(1, board_size + 1)]
        self.ship_sizes = (
            ship_sizes if ship_sizes else [5, 4, 3, 3, 2]
        )  # carrier, battleship, cruiser, sub, destroyer

        # Validate ship sizes are feasible for the board
        if any(size > self.board_size for size in self.ship_sizes):
            raise ValueError(
                "All ship sizes must be less than or equal to the board size"
            )

        # Instance-level RNG for reproducibility
        self.rng = random.Random(seed)

        self.reset()

    def reset(self, seed=None):
        # Optionally reseed for deterministic setup
        if seed is not None:
            self.rng.seed(seed)

        self.board = {f"{c}{r}": "?" for c in self.cols for r in self.rows}
        self.ships = []
        self.history = []
        self.turn_count = 0
        self.game_over = False
        self.last_move_invalid = False

        # Prepare ship names in the order of ship_sizes
        names_sequence = []
        default_sequence = [5, 4, 3, 3, 2]
        if self.ship_sizes == default_sequence:
            names_sequence = [
                "Carrier",
                "Battleship",
                "Cruiser",
                "Submarine",
                "Destroyer",
            ]
        else:
            seen_by_size = {}
            for sz in self.ship_sizes:
                seen_by_size[sz] = seen_by_size.get(sz, 0) + 1
                names_sequence.append(f"size-{sz} #{seen_by_size[sz]}")

        occupied = set()
        for idx, size in enumerate(self.ship_sizes):
            placed = False
            while not placed:
                horiz = self.rng.choice([True, False])
                if horiz:
                    row = self.rng.choice(self.rows)
                    start_idx = self.rng.randint(0, self.board_size - size)
                    coords = [f"{self.cols[start_idx + i]}{row}" for i in range(size)]
                else:
                    col = self.rng.choice(self.cols)
                    start_idx = self.rng.randint(0, self.board_size - size)
                    coords = [f"{col}{self.rows[start_idx + i]}" for i in range(size)]

                # check for overlap
                if not any(c in occupied for c in coords):
                    self.ships.append(
                        ShipStatus(
                            name=names_sequence[idx],
                            size=size,
                            coords=coords,
                            hits=set(),
                        )
                    )
                    for c in coords:
                        occupied.add(c)
                    placed = True

    def render(self) -> str:
        """
        Render the current board state as a string.
        """
        # Create header with column letters
        result = "  " + " ".join(self.cols) + "\n"

        # Add each row
        for row in self.rows:
            result += f"{row:>2}"
            for col in self.cols:
                cell = self.board[f"{col}{row}"]
                result += f" {cell}"
            result += "\n"

        return result

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
            if move in ship.coords:
                ship.hits.add(move)
                hit = True
                if ship.is_sunk:
                    sunk = True
                    sunk_ship = ship
                break

        if hit:
            if sunk:
                # Mark all positions of the sunk ship as 's'
                for coord in sunk_ship.coords:
                    self.board[coord] = "s"
            else:
                self.board[move] = "x"
        else:
            self.board[move] = "o"

        self.history.append(move)
        self.turn_count += 1

        # check for game over
        if all(ship.is_sunk for ship in self.ships):
            self.game_over = True

        return self.render(), hit, sunk, self.game_over, False

    def get_valid_moves(self):
        """
        Returns list of available (unknown) cells.
        """
        return [pos for pos, val in self.board.items() if val == "?"]

    def get_state(self) -> GameState:
        """
        Returns the current game state with full information.

        Returns:
            GameState object with complete game information.
        """
        board_state = BoardState(board_size=self.board_size, cells=self.board.copy())

        return GameState(
            turn_count=self.turn_count,
            game_over=self.game_over,
            last_move_invalid=self.last_move_invalid,
            history=self.history.copy(),
            board=board_state,
            ships=[
                ShipStatus(
                    name=ship.name,
                    size=ship.size,
                    coords=list(ship.coords),
                    hits=set(ship.hits),
                )
                for ship in self.ships
            ],
        )
