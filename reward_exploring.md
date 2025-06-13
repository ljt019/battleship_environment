# Rewards for GRPO

## win_reward_func
### Brief description
Reward for winning the game.

- **Parameters:**  
  - `completion`: List of message dicts (game transcript)  
  - `answer`: (unused)  
  - `**kwargs`: (unused)

- **Intended purpose:**  
  Reward the agent for winning the game.

- **Calculation details:**  
  Returns 1.0 if any user message contains "victory!", "you won!", or "all ships sunk" (case-insensitive). Otherwise, returns 0.0.

- **Range of values:**  
  0.0 or 1.0

- **Dependencies:**  
  Relies on user messages in the transcript containing specific victory phrases.

- **Known limitations or caveats:**  
  If the victory message is phrased differently, the reward may not trigger.

- **Example usage:**  
  Used as a main objective in the reward rubric.

- **Related rewards:**  
  None directly, but complements progress-based rewards.

---

## efficiency_reward_func
### Brief description
Reward for efficiency (fewer moves), applies to all games.

- **Parameters:**  
  - `completion`: List of message dicts  
  - `answer`: (unused)  
  - `**kwargs`: (unused)

- **Intended purpose:**  
  Encourage the agent to win in fewer moves.

- **Calculation details:**  
  Counts assistant moves. Applies exponential decay:  
  2^(-(moves-17)/10)  
  (17 moves = 1.0, 25 moves ≈ 0.57, 35 moves ≈ 0.30)

- **Range of values:**  
  [0.0, 1.0] – returns 1.0 for ≤17 moves, decays toward 0 as moves increase; returns 0.0 only if no assistant moves are recorded.

- **Dependencies:**  
  Number of assistant moves in the transcript.

- **Known limitations or caveats:**  
  Always active (even for losses). If no assistant moves are logged (edge-case), the reward is 0.0 which may slightly penalise initialisation steps.

- **Example usage:**  
  Used to encourage efficient play.

- **Related rewards:**  
  win_reward_func (complements by rewarding speed, not just victory).

---

## hit_reward_func
### Brief description
Reward for hitting ships.

- **Parameters:**  
  - `completion`: List of message dicts  
  - `answer`: (unused)  
  - `**kwargs`: (unused)

- **Intended purpose:**  
  Reward the agent for hitting ships.

- **Calculation details:**  
  For each user message containing "hit!" or "hit and sunk!" (but not "miss"), adds 0.1 to the reward.

- **Range of values:**  
  0.0 or positive multiples of 0.1 (e.g., 0.1, 0.2, ...)

- **Dependencies:**  
  User messages indicating hits.

- **Known limitations or caveats:**  
  Relies on consistent phrasing in user messages.

- **Example usage:**  
  Used to reward progress toward sinking ships.

- **Related rewards:**  
  sink_reward_func (for full sinks).

---

## sink_reward_func
### Brief description
Reward for sinking ships.

- **Parameters:**  
  - `completion`: List of message dicts  
  - `answer`: (unused)  
  - `**kwargs`: (unused)

- **Intended purpose:**  
  Reward the agent for sinking entire ships.

- **Calculation details:**  
  For each user message containing "hit and sunk!" or "destroyed an entire ship", adds 0.3 to the reward.

- **Range of values:**  
  0.0 or positive multiples of 0.3 (e.g., 0.3, 0.6, ...)

- **Dependencies:**  
  User messages indicating a ship was sunk.

- **Known limitations or caveats:**  
  Relies on specific phrases in user messages.

- **Example usage:**  
  Used to reward major progress in the game.

- **Related rewards:**  
  hit_reward_func (for partial progress).

---

## format_reward_func
### Brief description
Reward for proper move format.

- **Parameters:**  
  - `completion`: List of message dicts  
  - `answer`: (unused)  
  - `**kwargs`: (unused)

- **Intended purpose:**  
  Encourage the agent to use the correct move format.

- **Calculation details:**  
  For each assistant message, checks for the regex `<guess>\[[a-j][0-9]+\]</guess>`. Returns the fraction of assistant messages with valid format.

- **Range of values:**  
  [0.0, 1.0]

- **Dependencies:**  
  Assistant message formatting.

- **Known limitations or caveats:**  
  Only checks for a specific format; other valid formats are not rewarded.

- **Example usage:**  
  Used to enforce output formatting.

- **Related rewards:**  
  valid_move_reward_func (for move validity).

---

## valid_move_reward_func
### Brief description
Penalty for invalid moves (already played, out of bounds, etc.)

- **Parameters:**  
  - `completion`: List of message dicts  
  - `answer`: (unused)  
  - `**kwargs`: (unused)

- **Intended purpose:**  
  Penalize the agent for making invalid moves.

- **Calculation details:**  
  Counts user messages with "invalid move" or "invalid format". Counts assistant moves with valid guess format. Returns the fraction of valid moves:  
  max(0, (total moves - invalid count) / total moves)

- **Range of values:**  
  [0.0, 1.0]

- **Dependencies:**  
  User feedback and assistant move formatting.

- **Known limitations or caveats:**  
  Only penalizes moves flagged as invalid by user messages.

- **Example usage:**  
  Used to discourage repeated or out-of-bounds moves.

- **Related rewards:**  
  format_reward_func (for formatting), hit/sink rewards (for progress).

---

## Critique
### Strengths
- **Aligned with core objectives:** Rewards directly target key behaviours (winning, sinking ships, avoiding invalid moves).
- **Dense feedback signals:** Per-hit and per-sink rewards provide incremental shaping rather than only a sparse terminal reward.
- **Format & validity enforcement:** Separate rewards for move syntax and legality reduce derailment by keeping the conversation on-track.
- **Efficiency shaping:** Exponential decay on move count encourages faster victories without needing handcrafted per-turn penalties.

### Limitations / Pain Points
- **Phrase coupling:** All hit / sink / invalid detections rely on exact strings in environment messages. Any wording drift breaks reward capture.
- **Fixed scalar weights:** Current rubric weights (e.g., win=2.0, sink=1.0) were chosen heuristically; they may under- or over- emphasise certain behaviours as the agent improves.
- **Additive overlaps:** A single action can trigger multiple rewards (e.g., a "HIT AND SUNK!" counts for both hit and sink), which may distort relative importance.
- **Sparse win signal:** Victory reward still arrives only at the end; if training is unstable, agent may rarely experience it early on.
- **No negative reward for wasted turns:** Efficiency reward never drops below 0, so agents are not explicitly punished for very long games or repeated misses beyond the decay.
- **Valid-move metric blind to silent errors:** If the environment fails to flag an invalid move (edge case), the agent receives no penalty.

### Improvement Ideas
1. **Robust message parsing**  
   Use regex patterns tolerant to minor wording changes, or embed structured tags (e.g., `<result>hit</result>`) inside environment responses to make reward extraction reliable.
2. **Dynamic or learned weights**  
   Adapt weights during training based on reward sparsity or learning progress (e.g., increase win weight once hit/sink mastery is achieved).
3. **Negative efficiency term**  
   Shift efficiency reward to a symmetric scale (e.g., +1 for ≤17 moves down to −1 for ≥50 moves) so extremely long games incur an explicit penalty.
4. **Decouple hit vs. sink**  
   Consider removing the hit reward once the agent reaches a threshold proficiency, or normalise rewards so hit+sink does not double-count the same action.
5. **Introduce exploration bonuses**  
   Small positive reward for probing unexplored regions could mitigate early-game dithering.
6. **Rich invalid-move detection**  
   Track illegal coordinate patterns directly (board algebra) instead of relying solely on environment error strings.
7. **Curriculum scheduling**  
   Start with smaller boards or fewer ships and gradually scale difficulty, adjusting reward weights to maintain signal saturation.
8. **Reward normalisation/clipping**  
   Apply running normalisation to keep reward magnitudes comparable over time, stabilising PPO/GRPO updates.

### Open Questions
- What is the observed distribution of each reward during recent training—are some almost always zero or saturated?
- Do additive rewards lead to unintended optimisation of trivial behaviours (e.g., spamming valid but random moves to maximise format reward)?
- Would a sparse binary success/failure reward be sufficient once baseline competence is reached, simplifying the objective?