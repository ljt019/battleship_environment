import random
import string

from .base import GuessStrategy

class RandomGuessStrategy(GuessStrategy):
    def guess(self, _messages): 
        col = random.choice(string.ascii_lowercase[:10])
        row = random.randint(1, 10)
        return f"{col}{row}"