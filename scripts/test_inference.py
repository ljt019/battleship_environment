#!/usr/bin/env python3
"""Interactive viewer: watch the SFT model play Battleship turn-by-turn.

Usage: just run the script.  It loads the fine-tuned model with Transformers
(directly, no vLLM) and steps through a single game.  Press Enter to advance
to the next turn; Ctrl-C or 'exit' to quit.
"""

import re
import time
import threading
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, TextIteratorStreamer  # apply template + streaming

from verifiers.parsers.xml_parser import XMLParser

from src.battleship_grpo.battleship_env import (
    BattleshipEnv,
    BATTLESHIP_SYSTEM_PROMPT,
    BATTLESHIP_RULES,
)

# Local OpenAI wrapper
from src.local_openai import LocalOpenAI

# ----------------- CONFIG -------------------------------------------------- #
# Name or path of the model to load
#MODEL_NAME = "ljt019/Qwen3-1.7B-Battleship-SFT"
MODEL_NAME = "ljt019/Qwen3-1.7B-battleship-grpo"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
# device and dtype handled inside LocalOpenAI

# Instantiate parser and local client once
XML_PARSER = XMLParser(fields=["think", "guess"])
CLIENT = LocalOpenAI(MODEL_NAME)

def build_messages(system_prompt: str, board_compact: str, ships_block: str, history: List[Dict[str, str]]):
    """Return the messages list expected by Qwen chat template."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)  # previous conversation

    # Add the new user board message
    board_msg = (
        "Here's the current board:\n\n"
        f"<board>\n{board_compact}\n</board>\n\n"
        f"{ships_block}\n\nYour move:"
    )
    messages.append({"role": "user", "content": board_msg})
    return messages


def extract_think_and_guess(answer: str):
    """Return (think, coordinate) parsed from model answer."""
    # If the model omitted the opening <think> but included a closing tag, prepend it.
    if "<think>" not in answer.lower() and "</think>" in answer.lower():
        answer = "<think>" + answer

    parsed = XML_PARSER.parse(answer)
    think = getattr(parsed, "think", None) or "(no think)"
    guess_raw = getattr(parsed, "guess", None)
    coord = None
    if guess_raw:
        coord_match = re.search(r"([a-j](?:10|[1-9]))", guess_raw, re.IGNORECASE)
        if coord_match:
            coord = coord_match.group(1).lower()
    return think, coord


# For chat template only (prompt construction)
print(f"Loading tokenizer for {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ----------------- HELPER ---------------------------------------------- #

def chat_completion(messages: List[Dict[str, str]]) -> str:
    """Call the local OpenAI-compatible client and return assistant content."""
    response = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content  # type: ignore

# ----------------- STREAMING THINK --------------------------------------- #

def stream_think(messages: List[Dict[str, str]]) -> str:
    """Generate assistant reply locally and stream the contents inside the <think> block.

    Returns the full assistant answer string (reasoning + guess)."""

    # Get tokenizer/model from the LocalOpenAI cache (they are already loaded)
    tokenizer, model = CLIENT._get_model(MODEL_NAME)  # pylint: disable=protected-access

    # Build chat-template prompt text and tokens
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    gen_args = dict(
        **input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_args)
    thread.start()

    answer_buf = ""
    buffer = ""
    printing = False
    TAG_OPEN = "<think>"
    TAG_CLOSE = "</think>"

    for token in streamer:
        answer_buf += token
        buffer += token

        # Begin printing once we see <think>
        if not printing:
            start_idx = buffer.lower().find(TAG_OPEN)
            if start_idx != -1:
                buffer = buffer[start_idx + len(TAG_OPEN):]
                printing = True

        if printing:
            close_idx = buffer.lower().find(TAG_CLOSE)
            if close_idx == -1:
                # No closing tag yet – print everything and keep tail for overlap safety
                tail_keep = len(TAG_CLOSE) - 1
                if len(buffer) > tail_keep:
                    print(buffer[:-tail_keep], end="", flush=True)
                    buffer = buffer[-tail_keep:]
            else:
                # Print up to </think> then stop streaming output
                print(buffer[:close_idx], flush=True)
                # Remove printed segment + tag from buffer
                buffer = buffer[close_idx + len(TAG_CLOSE):]
                printing = False  # ignore rest of tokens for console

    thread.join()
    return answer_buf.strip()

def main():
    # tokenizer already loaded globally

    # Instantiate the training environment (for its board logic and parsing)
    env = BattleshipEnv(num_samples=0, num_eval_samples=0)

    # Conversation seed: rules only. System prompt is handled separately.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": BATTLESHIP_SYSTEM_PROMPT},
        {"role": "user", "content": BATTLESHIP_RULES},
    ]

    # Obtain initial board state from the environment
    state: Dict[str, Any] = {}
    env_msg, state = env.env_response(messages, state)
    messages.append(env_msg)

    # Keep a reference to the immutable rules message (index 1)
    rules_msg = messages[1]

    while True:
        # Compact history but always keep the full rules message intact
        non_rule_msgs = [m for m in messages if m not in (messages[0], rules_msg)]
        non_rule_msgs = env._compact_conversation_history(non_rule_msgs)
        messages = [messages[0], rules_msg] + non_rule_msgs

        # Print reasoning (no streaming):
        print("\nModel reasoning:")
        answer = stream_think(messages)

        # Parse reasoning/coordinate from full answer
        think, coord = extract_think_and_guess(answer)

        if think != "(no think)":
            print("\nModel reasoning:", think)

        if coord:
            print("\nModel Turn: [", coord, "]")

        # Append assistant reply
        messages.append({"role": "assistant", "content": answer})

        # Let environment compute result / next board
        env_msg, state = env.env_response(messages, state)
        messages.append(env_msg)

        # Pretty-print board section
        board_match = re.search(r"<board>\s*(.*?)\s*</board>", env_msg["content"], re.DOTALL)
        if board_match:
            board_compact = board_match.group(1).strip()
            from src.battleship_grpo.battleship_game import BattleshipGame  # local import to avoid circular
            print("\n",BattleshipGame.compact_to_pretty(board_compact), "\n")

        # Print result line (first line before <board>)
        result_line = env_msg["content"].split("\n", 1)[0]
        print(result_line)

        if env.is_completed(messages, state):
            print("\n🎉 Game over!", result_line)
            break

        # Wait user
        inp = input("\nPress Enter for next move, or type 'exit': ")
        if inp.strip().lower() in {"exit", "quit"}:
            print("--------------------------------")
            print("Final messages")
            print("--------------------------------")
            # print all the messages in a pretty format
            for msg in messages:
                print(f"{msg['role']}: {msg['content']}")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
