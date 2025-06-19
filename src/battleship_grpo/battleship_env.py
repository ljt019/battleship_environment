import random
from typing import Tuple, List, Dict, Any, Union
import re
from copy import deepcopy

from datasets import Dataset
from openai import OpenAI

from verifiers import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from .battleship_game import BattleshipGame
from .rewards import setup_reward_rubric

BATTLESHIP_SYSTEM_PROMPT = """You are playing Battleship.

After every turn, the environment sends ONE user message containing the current game state in tagged form:

<result move="c3" value="hit|miss|sunk|invalid|victory"/>
<remaining carrier="N" battleship="N" cruiser="N" submarine="N" destroyer="N"/>
<state hits="a5 e4" misses="b1 d6" sunk="d5 e5" unknown="83"/>
<grid>
(? unknown, o miss, x hit, s sunk)
10x10 grid representing current board state
</grid>

Rules for you:
1. Inside <think>, reference ONLY coordinates appearing in the hits, misses, or sunk lists.
2. Finish ships by guessing cells directly adjacent (up, down, left, right—no diagonals) to confirmed hits before exploring new areas.
3. Keep <think> ≤ 75 tokens.
4. Respond EXACTLY in the following format and nothing else:

<think>
Concise reasoning about the next best shot.
</think>

<guess>[coordinate]</guess>"""

BATTLESHIP_RULES = """Goal
———
Sink all enemy ships by guessing coordinates.

Coordinate format
  - Column letters (a-j) + row numbers (1-10), e.g., e5.

Symbols in <grid>
  ? unknown   o miss   x hit (unsunk)   s sunk-ship part

Per-turn tags (sent each turn)
  - <result move="c3" value="hit|miss|sunk|invalid|victory"/> outcome of your last shot
  - <remaining carrier="…" …/> ships still afloat
  - <state hits="…" misses="…" sunk="…" unknown="N"/> status of guessed cells
  - <grid> header line + 10 rows </grid> current board representation

Ship sizes
  Carrier (5) • Battleship (4) • Cruiser (3) • Submarine (3) • Destroyer (2)

Important rules
  - NEVER guess a cell that isn't marked "?" (unknown) on the grid.
  - Guessing previously guessed cells (marked o, x, or s) is invalid."""

class BattleshipEnv(MultiTurnEnv):
    """
    Battleship environment for GRPO training.
    """
    def __init__(self,
                 num_samples: int = 1000,
                 num_eval_samples: int = 100,
                 seed: int = 0,
                 **kwargs):
        self.num_samples = num_samples
        self.num_eval_samples = num_eval_samples
        self.seed = seed
        dataset, eval_dataset = self.create_hf_datasets()
        parser = XMLParser(fields=["think", "guess"], answer_field="guess")
        rubric = Rubric(parser=parser)
        
        rubric = setup_reward_rubric(rubric)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=BATTLESHIP_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type='chat',
            **kwargs
        )

    def is_completed(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        """Check if the battleship game is completed"""
        if 'game' not in state:
            return False
        
        game = state['game']
        return game.game_over

    def env_response(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate environment response to player move"""
        # Initialize game if needed
        if 'game' not in state:
            game = BattleshipGame()
            game.reset()  # This places ships randomly
            state['game'] = game
            
            # Return initial board state
            board_render = game.render_compact()

            ships_block = self._build_ships_remaining(game)
            state_tag = self._build_state_tag(game)
            remaining_tag = self._build_remaining_tag(game)
            grid_str = self._build_ascii_grid(game)
            content = (
                "Here's the starting board:\n\n" +
                remaining_tag + "\n" +
                state_tag + "\n" +
                "<grid>\n" + grid_str + "\n</grid>\n\n" +
                "Next move:"
            )
            return {"role": "user", "content": content}, state
        
        game = state['game']
        
        # Parse the last assistant message for the move
        last_message = messages[-1]["content"]
        
        # Extract coordinate from <guess>[coordinate]</guess> (rows 1-10 only)
        coord_match = re.search(r'<guess>\[([a-j](?:10|[1-9]))\]</guess>', last_message, re.IGNORECASE)
        if not coord_match:
            # Fallback: look for any coordinate in brackets
            coord_match = re.search(r'\[([a-j](?:10|[1-9]))\]', last_message, re.IGNORECASE)
        
        if not coord_match:
            # If assistant gave bad format, remind and resend current board & ships list
            board_render = game.render_compact()
            ships_block = self._build_ships_remaining(game)

            content = (
                "INVALID FORMAT! Please use: <guess>[coordinate]</guess> (e.g., <guess>[e5]</guess>)\n\n"
                f"<board>\n{board_render}\n</board>\n\n" +
                ships_block + "\n\nNext move:"
            )
            return {"role": "user", "content": content}, state
        
        coordinate = coord_match.group(1).lower()
        
        # step game forwad
        board_render, hit, sunk, game_over, invalid = game.step(coordinate)
        
        board_render = game.render_compact()
        state['game'] = game
        
        # Generate result message
        if invalid:
            result = "invalid"
        elif hit:
            if sunk:
                result = "sunk"
            else:
                result = "hit"
        else:
            result = "miss"
        
        ships_block = self._build_ships_remaining(game)
        state_tag = self._build_state_tag(game)
        remaining_tag = self._build_remaining_tag(game)
        grid_str = self._build_ascii_grid(game)

        # Generate response
        if game_over:
            content = (
                f"<result move=\"{coordinate}\" value=\"victory\"/>\n"
                f"{remaining_tag}\n{state_tag}\n"
                f"<grid>\n{grid_str}\n</grid>\n\nVICTORY! You sunk all ships in {game.turn_count} moves!"
            )
        else:
            content = (
                f"<result move=\"{coordinate}\" value=\"{result}\"/>\n"
                f"{remaining_tag}\n{state_tag}\n"
                f"<grid>\n{grid_str}\n</grid>\n\nNext move:"
            )
        
        return {"role": "user", "content": content}, state
    
    def _compact_conversation_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove old board states from conversation, keeping only the most recent one"""
        compacted_messages = []
        last_board_message_idx = -1
        
        # Find the last user message containing a <grid> tag (latest board)
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user' and '<grid>' in msg.get('content', ''):
                last_board_message_idx = i
        
        # Keep all messages, but replace old board states with just the feedback
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                is_env_state = any(tag in content for tag in ('<grid>', '<state', '<remaining', '<result'))
                # If this is an old environment state (not the most recent), drop it entirely
                if is_env_state and i < last_board_message_idx:
                    continue  # skip stale env message
                else:
                    compacted_messages.append(msg)
            else:
                compacted_messages.append(msg)
        
        return compacted_messages
    
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment, with conversation compacting.
        """
        is_completed = False
        state = {'answer': answer}
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        completion = []
        turn = 0
        
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            
            if 'game' not in state:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
            
            # Compact the conversation history so that only the most recent
            # board state is preserved *in place*. This keeps the prompt
            # length bounded **and** guarantees that the tokens seen by the
            # model match the tokens later processed by `process_chat_format`,
            # preventing KL spikes due to token-mismatch.

            messages = self._compact_conversation_history(messages)

            response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state
    
    def create_hf_datasets(self) -> Tuple[Dataset, Dataset]:
        """Create HuggingFace datasets for training and evaluation"""
        dataset_rows = []
        eval_dataset_rows = []
        
        # Set seed for reproducibility
        random.seed(self.seed)
        
        for i in range(self.num_samples + self.num_eval_samples):
            question = BATTLESHIP_RULES
            # don't have a single "answer"
            # The answer is determined by random ship placement during gameplay
            answer = f"game_{i}" 
            
            if i < self.num_samples:
                dataset_rows.append({
                    "question": question,
                    "answer": answer
                })
            else:
                eval_dataset_rows.append({
                    "question": question,
                    "answer": answer
                })
        
        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_dataset_rows)
        return dataset, eval_dataset 

    def _build_ships_remaining(self, game: "BattleshipGame") -> str:
        """Return a formatted string listing all ships that are not yet sunk."""
        remaining_lines: List[str] = []
        ship_counts: Dict[int, int] = {}
        for ship in game.ships:
            if set(ship["hits"]) != set(ship["coords"]):
                size = len(ship["coords"])
                # Cruiser and Submarine are both size 3 – distinguish the first as Cruiser
                if size == 3:
                    ship_counts[3] = ship_counts.get(3, 0) + 1
                    name = "Cruiser" if ship_counts[3] == 1 else "Submarine"
                else:
                    name = game.ship_names.get(size, f"{size}")
                remaining_lines.append(f" {name.ljust(10)} {size}")
        if not remaining_lines:
            remaining_lines.append(" None")
        return "Ships remaining:\n" + "\n".join(remaining_lines) 

    def _build_state_tag(self, game: "BattleshipGame") -> str:
        """Return a <state/> tag listing hits, misses, sunk squares and unknown count."""
        hits: list[str] = []
        misses: list[str] = []
        sunk: list[str] = []

        for coord, val in game.board.items():
            if val == "x":
                hits.append(coord)
            elif val == "o":
                misses.append(coord)
            elif val == "s":
                sunk.append(coord)

        unknown_count = sum(1 for v in game.board.values() if v == "?")

        hits_str = " ".join(sorted(hits))
        misses_str = " ".join(sorted(misses))
        sunk_str = " ".join(sorted(sunk))
        return (
            f"<state hits=\"{hits_str}\" "
            f"misses=\"{misses_str}\" "
            f"sunk=\"{sunk_str}\" "
            f"unknown=\"{unknown_count}\"/>"
        )

    def _build_remaining_tag(self, game: "BattleshipGame") -> str:
        """Return <remaining …/> tag with counts for each ship type."""
        counts = {"carrier": 0, "battleship": 0, "cruiser": 0, "submarine": 0, "destroyer": 0}
        size3_seen = 0
        for ship in game.ships:
            if set(ship["hits"]) != set(ship["coords"]):
                size = len(ship["coords"])
                if size == 5:
                    counts["carrier"] += 1
                elif size == 4:
                    counts["battleship"] += 1
                elif size == 3:
                    size3_seen += 1
                    if size3_seen == 1:
                        counts["cruiser"] += 1
                    else:
                        counts["submarine"] += 1
                elif size == 2:
                    counts["destroyer"] += 1
        attrs = " ".join(f"{k}=\"{v}\"" for k, v in counts.items())
        return f"<remaining {attrs} />"

    def _build_ascii_grid(self, game: "BattleshipGame") -> str:
        """Return ASCII grid string with header and rows."""
        cols = game.cols  # 'abcdefghij'
        header = " " + " ".join(cols)
        lines = [header]
        for r in game.rows:
            row_cells = []
            for c in cols:
                val = game.board[f"{c}{r}"]
                row_cells.append(val)
            lines.append(f"{r.rjust(2)} {' '.join(row_cells)}")
        return "\n".join(lines)

    def process_chat_format(self, prompt, completion, processing_class, mask_env_responses: bool = False):
        """Override the parent `process_chat_format` to tolerate chat-history
        divergence. We first attempt the standard incremental-prefix path from
        the superclass. If it raises an `AssertionError` (token prefix
        mismatch), we recompute the tokenisation from scratch while preserving
        the same masking semantics.
        """
        try:
            # Fast path – just delegate to the parent implementation
            return super().process_chat_format(prompt, completion, processing_class, mask_env_responses)
        except AssertionError:
            # Slow path – rebuild tokenisation without the strict prefix check
            prompt_text = processing_class.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            assert isinstance(prompt_text, str)
            prompt_ids = processing_class.encode(prompt_text)
            prompt_mask = [1] * len(prompt_ids)

            completion_ids = []
            completion_mask = []
            prev_ids = prompt_ids

            for i, msg in enumerate(completion):
                conversation_prefix = prompt + completion[: i + 1]
                prefix_text = processing_class.apply_chat_template(
                    conversation_prefix, tokenize=False, add_generation_prompt=False
                )
                assert isinstance(prefix_text, str)
                current_ids = processing_class.encode(prefix_text)
                new_tokens = current_ids[len(prev_ids) :]
                completion_ids.extend(new_tokens)

                # Apply the same masking rules as the parent implementation
                if msg["role"] == "assistant":
                    msg_mask = [1] * len(new_tokens)
                elif msg["role"] != "assistant" and mask_env_responses:
                    msg_mask = [0] * len(new_tokens)
                else:
                    msg_mask = [1] * len(new_tokens)
                completion_mask.extend(msg_mask)
                prev_ids = current_ids

            return prompt_ids, prompt_mask, completion_ids, completion_mask 