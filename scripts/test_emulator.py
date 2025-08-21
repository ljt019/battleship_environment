# /// script
# dependencies = [
#   "textual",
#   "rich",
# ]
# ///

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from battleship_emulator import BattleshipEmulator, GameState


def render_board_table(state: GameState) -> Table:
    table = Table(show_header=True, show_lines=True)
    table.add_column("", style="bold", width=2)

    # Add column headers (A-J for 10x10)
    for i in range(state.board.board_size):
        col_letter = chr(ord("A") + i)
        table.add_column(col_letter, justify="center", width=3)

    # Add rows
    for row in range(1, state.board.board_size + 1):
        row_data = [str(row)]
        for col in range(state.board.board_size):
            col_letter = chr(ord("a") + col).lower()
            cell_key = f"{col_letter}{row}"
            cell_value = state.board.cells[cell_key]

            if cell_value == "?":
                row_data.append("[dim]?[/dim]")
            elif cell_value == "o":
                row_data.append("[cyan]○[/cyan]")
            elif cell_value == "x":
                row_data.append("[yellow]●[/yellow]")
            elif cell_value == "s":
                row_data.append("[green bold]■[/green bold]")
            else:
                row_data.append(cell_value)

        table.add_row(*row_data)

    return table


def summarize_board(state: GameState) -> dict:
    hits = sum(1 for v in state.board.cells.values() if v in ["x", "s"])
    misses = sum(1 for v in state.board.cells.values() if v == "o")
    unknowns = len(state.board.unknown_cells)
    return {"hits": hits, "misses": misses, "unknowns": unknowns}


class BoardView(Static):
    def update_board(self, state: GameState) -> None:
        table = render_board_table(state)
        self.update(table)


class StatusView(Static):
    def update_status(self, state: GameState, last_result: dict | None = None) -> None:
        summary = summarize_board(state)
        sunk_ships = [ship.name for ship in state.ships if ship.is_sunk]

        status_text = f"Turn: {state.turn_count}\n"
        status_text += f"Game Over: {'Yes' if state.game_over else 'No'}\n"
        status_text += f"Hits: {summary['hits']} | Misses: {summary['misses']}\n"
        status_text += f"Unknown: {summary['unknowns']}\n\n"

        if last_result:
            move = last_result["move"]
            if last_result["invalid"]:
                status_text += f"Last: {move} - [red]Invalid[/red]\n"
            elif last_result["sunk"]:
                status_text += f"Last: {move} - [green bold]Hit & Sunk![/green bold]\n"
            elif last_result["hit"]:
                status_text += f"Last: {move} - [yellow]Hit[/yellow]\n"
            else:
                status_text += f"Last: {move} - [cyan]Miss[/cyan]\n"

        if sunk_ships:
            status_text += "\nSunk Ships:\n"
            for ship in sunk_ships:
                status_text += f"• {ship}\n"

        if state.game_over:
            status_text += (
                "\n[green bold]Victory![/green bold]\n[dim]Press Enter to exit[/dim]"
            )

        self.update(status_text)


class AutoPlayView(Static):
    def update_status(self, playing: bool, sleep_time: float) -> None:
        if playing:
            text = "[bold]Auto-playing...[/bold]\n"
            text += f"Speed: {sleep_time:.1f}s between moves\n"
            text += "[dim]Press 'p' to pause[/dim]"
        else:
            text = "[yellow]Paused[/yellow]\n"
            text += "[dim]Press 'p' to resume[/dim]"
        self.update(text)


class BattleshipTestApp(App):
    CSS = """
    #board-container {
        width: 50%;
        padding: 1;
    }

    #status-container {
        width: 30%;
        padding: 1;
    }

    #autoplay-container {
        width: 20%;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reset", "Reset"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("enter", "exit_if_done", "Exit (when done)"),
    ]

    def __init__(
        self,
        board_size: int = 10,
        seed: int | None = None,
        sleep_time: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.board_size = board_size
        self.seed = seed
        self.sleep_time = sleep_time
        self.emulator = None
        self.last_result = None
        self.is_playing = True
        self.game_task = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal():
                with Vertical(id="board-container"):
                    yield Static("## Battleship Board", id="board-title")
                    yield BoardView(id="board")
                with Vertical(id="status-container"):
                    yield Static("## Game Status", id="status-title")
                    yield StatusView(id="status")
                with Vertical(id="autoplay-container"):
                    yield Static("## Auto-Play", id="autoplay-title")
                    yield AutoPlayView(id="autoplay")

    def on_mount(self) -> None:
        self.emulator = BattleshipEmulator(self.board_size, seed=self.seed)
        self.refresh_ui()
        self.start_autoplay()

    def refresh_ui(self) -> None:
        state = self.emulator.get_state()
        self.query_one("#board", BoardView).update_board(state)
        self.query_one("#status", StatusView).update_status(state, self.last_result)
        self.query_one("#autoplay", AutoPlayView).update_status(
            self.is_playing, self.sleep_time
        )

    def generate_random_move(self) -> str:
        """Generate a random valid move."""
        state = self.emulator.get_state()
        unknown_cells = list(state.board.unknown_cells)
        if unknown_cells:
            return random.choice(unknown_cells)
        return "a1"  # Fallback

    async def autoplay_game(self) -> None:
        """Auto-play the game with random moves."""
        while not self.emulator.get_state().game_over:
            if not self.is_playing:
                await asyncio.sleep(0.1)
                continue

            move = self.generate_random_move()
            render, hit, sunk, game_over, invalid = self.emulator.step(move)
            self.last_result = {
                "move": move,
                "hit": hit,
                "sunk": sunk,
                "invalid": invalid,
            }
            self.refresh_ui()

            await asyncio.sleep(self.sleep_time)

        # Game is over, update UI one final time
        self.refresh_ui()

    def start_autoplay(self) -> None:
        """Start the autoplay task."""
        if self.game_task is None or self.game_task.done():
            self.game_task = asyncio.create_task(self.autoplay_game())

    def action_reset(self) -> None:
        # Cancel current game
        if self.game_task:
            self.game_task.cancel()

        self.emulator.reset()
        self.last_result = None
        self.is_playing = True
        self.refresh_ui()
        self.start_autoplay()

    def action_toggle_pause(self) -> None:
        self.is_playing = not self.is_playing
        self.refresh_ui()

    def action_exit_if_done(self) -> None:
        if self.emulator.get_state().game_over:
            self.exit()

    def action_quit(self) -> None:
        if self.game_task:
            self.game_task.cancel()
        self.exit()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-playing Battleship Emulator Test"
    )
    parser.add_argument("--size", type=int, default=10, help="Board size (default: 10)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible games")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Seconds between moves (default: 1.0)"
    )

    args = parser.parse_args()

    app = BattleshipTestApp(board_size=args.size, seed=args.seed, sleep_time=args.speed)
    app.run()


if __name__ == "__main__":
    main()
