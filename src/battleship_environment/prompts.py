"""
Prompt definitions for the Battleship environment.
"""

BATTLESHIP_SYSTEM_PROMPT = """
You are playing Battleship.

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

<guess>[coordinate]</guess>
"""

BATTLESHIP_INITIAL_MESSAGE = """
Goal
 - Sink all enemy ships by guessing coordinates.

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
  - Guessing previously guessed cells (marked o, x, or s) is invalid.

<result move="" value="start"/>
<remaining carrier="1" battleship="1" cruiser="1" submarine="1" destroyer="1" />
<state hits="" misses="" sunk="" unknown="100"/>
<grid>
   a b c d e f g h i j
 1 ? ? ? ? ? ? ? ? ? ?
 2 ? ? ? ? ? ? ? ? ? ?
 3 ? ? ? ? ? ? ? ? ? ?
 4 ? ? ? ? ? ? ? ? ? ?
 5 ? ? ? ? ? ? ? ? ? ?
 6 ? ? ? ? ? ? ? ? ? ?
 7 ? ? ? ? ? ? ? ? ? ?
 8 ? ? ? ? ? ? ? ? ? ?
 9 ? ? ? ? ? ? ? ? ? ?
10 ? ? ? ? ? ? ? ? ? ?
</grid>

Next move:
"""
