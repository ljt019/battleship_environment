import re
from typing import Callable
import verifiers as vf

# Model will respond like '<think> thinking about answer... </think> [b6]'

class BattleshipAnswerParser(vf.Parser):
    def parse_answer(self, completion):
        """Extract the answer from square brackets like [b6]"""
        if isinstance(completion, str):
            text = completion
        else:
            # Get the last assistant message
            text = completion[-1]["content"] if completion else ""
        
        # Extract content from square brackets
        match = re.search(r'\[([^\]]+)\]', text)
        return match.group(1) if match else None
    
    def parse(self, text: str):
        """Parse the text and return the content from square brackets"""
        match = re.search(r'\[([^\]]+)\]', text)
        return match.group(1) if match else None
    
    def get_format_str(self) -> str:
        """Return a string that describes the expected format."""
        return "Answer should be provided in square brackets, e.g., [b6]"
    
    def format(self, answer: str) -> str:
        """Format an answer into the expected square bracket format."""
        return f"[{answer}]"
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks if messages follow the expected format."""
        def format_reward_func(completion, **kwargs) -> float:
            """Reward function that checks if the response contains properly formatted square brackets."""
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0
            
            # Check the last message for proper formatting
            last_message = model_messages[-1]['content']
            
            # Check if there's content in square brackets
            bracket_match = re.search(r'\[([^\]]+)\]', last_message)
            if bracket_match:
                # Additional checks for valid battleship coordinates
                answer = bracket_match.group(1).lower()
                # Check if it's a valid battleship coordinate (letter + number)
                if re.match(r'^[a-j]([1-9]|10)$', answer):
                    return 1.0
                else:
                    return 0.7  # Has brackets but invalid coordinate
            else:
                return 0.0
        
        return format_reward_func