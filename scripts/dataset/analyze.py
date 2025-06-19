"""
Analyze which reward components dominate the per-turn score
===========================================================

For each example in ljt019/battleship-sft-new-format-patched we:
  • reconstruct (prompt, completion) as a chat list
  • run the BattleshipEnv rubric on that single turn
  • collect the raw contribution of every reward function
Finally we print aggregate stats and show the worst-offending component.
"""

from datasets import load_dataset
from collections import defaultdict, Counter
import statistics, tqdm, re, os, json
import asyncio
import numpy as np

from src.battleship_grpo import BattleshipEnv
from src.battleship_grpo.battleship_env import BATTLESHIP_SYSTEM_PROMPT

DATASET = "ljt019/battleship-sft-new-format-patched"
SPLIT   = "train"     # change if needed

# -------------------------------------------------------------
# 1. load dataset & env
# -------------------------------------------------------------
ds = load_dataset(DATASET, split=SPLIT)
env = BattleshipEnv(max_turns=1)      # we'll only call its rubric

rubric = env.rubric  # verifiers.rubrics.Rubric object

# Support both (func, weight) tuples and bare functions
def _extract_func(obj):
    if callable(obj):
        return obj
    if isinstance(obj, (list, tuple)) and callable(obj[0]):
        return obj[0]
    raise TypeError("Unexpected reward func entry type")

reward_funcs = [ _extract_func(item).__name__ for item in rubric.reward_funcs ]

# convenient map name -> running list of values
comp_stats = defaultdict(list)

# regex to strip the static system+rules so the reward functions
# don't get confused by duplicated text
sys_regex = re.escape(BATTLESHIP_SYSTEM_PROMPT)

# extra diagnostics for follow-up reward
followup_counts = []  # number of follow-up bonuses per game
num_moves_list = []   # assistant moves per game

# -------------------------------------------------------------
# 2. group turns into full games and score each game
# -------------------------------------------------------------

# helper to extract unknown attribute from a board message
_unknown_re = re.compile(r'unknown="(\d+)"', re.I)

games = defaultdict(list)  # game_id -> list of (unknown_count:int, example dict)

# gather turns per game
for ex in ds:
    game_id = ex.get("answer", None) or "unknown_game"
    board_msg = ex["prompt"][-1]  # last message in prompt is the current board state
    m = _unknown_re.search(board_msg.get("content", ""))
    unknown_cnt = int(m.group(1)) if m else -1  # -1 if missing
    games[game_id].append((unknown_cnt, ex))


async def _score(prompt_msgs, answer_text):
    """Call rubric.score_rollout (async) and return dict of component scores."""
    return await rubric.score_rollout(
        prompt=prompt_msgs,
        completion=prompt_msgs,
        answer=answer_text,
        state={},
        task=None,
    )


for game_id, turns in tqdm.tqdm(games.items(), desc="scoring games"):
    # sort by unknown descending so we go from first -> last turn
    turns_sorted = sorted(turns, key=lambda t: t[0], reverse=True)

    if not turns_sorted:
        continue

    # initial static msgs from first turn's prompt
    first_ex = turns_sorted[0][1]
    system_msg, rules_msg = first_ex["prompt"][0], first_ex["prompt"][1]
    convo = [system_msg, rules_msg]

    for _, ex in turns_sorted:
        board_msg = ex["prompt"][-1]
        asst_msg = ex["completion"][0]
        convo.extend([board_msg, asst_msg])

    scores = asyncio.run(_score(convo, game_id))

    for name in reward_funcs:
        comp_stats[name].append(scores.get(name, 0.0))

    # diagnostics
    fu_reward = scores.get('follow_up_reward_func', 0.0)
    followup_counts.append(round(fu_reward / 0.15))
    num_moves = len([m for m in convo if m['role'] == 'assistant'])
    num_moves_list.append(num_moves)

# replace dataset length for per-game stats
num_games = len(games)

# -------------------------------------------------------------
# 3. aggregate & display
# -------------------------------------------------------------
print("\nComponent-wise statistics (mean, min, max):")
table = []
for name in reward_funcs:
    vals = comp_stats[name]
    table.append((statistics.mean(vals), name,
                  min(vals), max(vals)))
table.sort()           # ascending by mean

for mean_v, name, mn, mx in table:
    print(f"{name:25}  mean={mean_v:+.3f}   min={mn:+.3f}   max={mx:+.3f}")

# identify biggest negative contributor per sample
worst_counter = Counter()
for i in range(num_games):
    worst_name = min(
        ((name, comp_stats[name][i]) for name in reward_funcs),
        key=lambda kv: kv[1]
    )[0]
    worst_counter[worst_name] += 1

print("\nMost common worst-offending component across games:")
for name, cnt in worst_counter.most_common():
    print(f"{name:25}: {cnt}  ({cnt/num_games*100:.1f} %)")

# -------------------------------------------------------------
# 4. diagnostics for follow-up reward
# -------------------------------------------------------------

if followup_counts:
    ratios = [f / n if n else 0 for f, n in zip(followup_counts, num_moves_list)]
    print("\nFollow-up diagnostics:")
    print(f"   Mean follow-up bonuses per game: {np.mean(followup_counts):.2f}")
    print(f"   Mean moves per game           : {np.mean(num_moves_list):.1f}")
    print(f"   Mean fraction follow-up moves : {np.mean(ratios)*100:.1f}%")
    print(f"   Max fraction follow-up moves  : {max(ratios)*100:.1f}%")

    # show top 5 games with highest follow-up ratio
    top_idx = np.argsort(ratios)[-5:][::-1]
    print("\nTop 5 games by follow-up ratio:")
    for idx in top_idx:
        print(f"   Game #{idx}: follow-ups={followup_counts[idx]} of {num_moves_list[idx]} moves ({ratios[idx]*100:.1f}%)")

# optional: dump JSON with full per-component arrays for deeper analysis
OUT = "reward_component_stats.json"
with open(OUT, "w") as f:
    json.dump(comp_stats, f)
print(f"\nWrote raw component arrays to {OUT}")