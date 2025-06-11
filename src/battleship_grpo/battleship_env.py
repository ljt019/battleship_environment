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

BATTLESHIP_SYSTEM_PROMPT = """You are playing a game of Battleship. The current board will be provided inside <board> … </board> tags as a compact grid (described in the rules). Use it to plan and execute your next shot. 

Respond ONLY in the following format:
<think>
Concise reasoning about what move to take next.
</think>

<guess>[coordinate]</guess>"""

BATTLESHIP_RULES = """Your goal is to sink all enemy ships by guessing their locations.

Board format (inside <board> tags):
  • First line:  a-j column letters (e.g.  abcdefghij )
  • Following lines: row number (1-10) immediately followed by ten single-character cells.
    ? unknown   x hit   o miss   s sunk-ship part

Strategy tips:
- After a hit (x), probe adjacent squares to locate the full ship.
- Use probability/spacing to target unexplored areas.
- Ship sizes: Carrier(5), Battleship(4), Cruiser(3), Submarine(3), Destroyer(2).

Make your next strategic move to sink all ships efficiently."""


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
            content = (
                "Here's the starting board:\n\n"
                "<board>\n" + board_render + "\n</board>\n\n" +
                ships_block + "\n\n" +
                "You are the player. Make your first move:"
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
            result = "INVALID MOVE - That square was already guessed or doesn't exist!"
        elif hit:
            if sunk:
                result = "HIT AND SUNK! You destroyed an entire ship!"
            else:
                result = "HIT! You found part of a ship - try adjacent squares!"
        else:
            result = "MISS - Only water here."
        
        ships_block = self._build_ships_remaining(game)

        # Generate response
        if game_over:
            content = (
                f"{result}\n\n<board>\n{board_render}\n</board>\n\n"
                f"{ships_block}\n\nVICTORY! You sunk all ships in {game.turn_count} moves!"
            )
        else:
            content = (
                f"{result}\n\n<board>\n{board_render}\n</board>\n\n"
                f"{ships_block}\n\nNext move:"
            )
        
        return {"role": "user", "content": content}, state
    
    def _compact_conversation_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove old board states from conversation, keeping only the most recent one"""
        compacted_messages = []
        last_board_message_idx = -1
        
        # Find the last user message containing a <board> tag
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user' and '<board>' in msg.get('content', ''):
                last_board_message_idx = i
        
        # Keep all messages, but replace old board states with just the feedback
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # If this is an old board state (not the most recent), strip the <board> block
                if '<board>' in content and i < last_board_message_idx:
                    lines = content.split('\n')
                    feedback_line = lines[0] if lines else content
                    compacted_content = f"{feedback_line}\nNext move:"
                    compacted_messages.append({"role": "user", "content": compacted_content})
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
            
            # Feed the full conversation history (without additional compaction) to ensure
            # that the tokens seen by the model match exactly the tokens that will later
            # be processed by the verifiers library. This prevents chat-format
            # tokenization mismatches detected by `process_chat_format`.
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

    def process_chat_format(self, messages: List[Dict[str, Any]], prev_ids=None, **kwargs):
        """Wrapper around the parent `process_chat_format` that gracefully handles
        occasional token-prefix mismatches.

        The upstream implementation asserts that the newly tokenised chat history
        must start with the exact token sequence produced on the previous call.
        This assumption can be violated when earlier user messages are trimmed
        or otherwise modified (e.g. board compaction) between calls, which would
        raise an `AssertionError` and kill the training loop.

        We keep the same behaviour when the assertion passes, but if a mismatch
        is detected we simply fall back to re-tokenising from scratch (i.e. we
        ignore the cached `prev_ids`). This is safe because the returned token
        IDs and masks will still correspond to the *current* chat transcript,
        which is what the trainer consumes.
        """
        try:
            # Try the standard incremental path first – this is more efficient
            return super().process_chat_format(messages, prev_ids, **kwargs)
        except AssertionError:
            # Fallback: regenerate without comparing to previous ids
            return super().process_chat_format(messages, None, **kwargs) 