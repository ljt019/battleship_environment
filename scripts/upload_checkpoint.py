#!/usr/bin/env python3
"""Upload a Battleship-GRPO checkpoint directory to the Hugging Face Hub.

Usage
-----
$ python scripts/upload_checkpoint.py \
    --ckpt_dir outputs/battleship-grpo-1.7b \
    --repo_id ljt019/Qwen3-1.7B-Battleship-GRPO \
    [--private] [--token YOUR_HF_TOKEN]

If `--token` is omitted the script falls back to the `HF_TOKEN` environment
variable (recommended).  Re-runs only push changed files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder, upload_folder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a checkpoint to HuggingFace Hub")
    p.add_argument("--ckpt_dir", required=True, help="Local checkpoint folder (must contain config.json, model*.bin, tokenizer files, …)")
    p.add_argument("--repo_id", required=True, help="Destination repo id, e.g. ljt019/Qwen3-1.7B-Battleship-GRPO")
    p.add_argument("--token", default=None, help="HF access token; fallback to HF_TOKEN env var or cached login")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    return p.parse_args()


def ensure_token(token: Optional[str] = None) -> str:
    token = token or os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        sys.exit("[ERROR] No HuggingFace token provided. Pass --token or set HF_TOKEN env var or run `huggingface-cli login`. ")
    return token


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt_dir).expanduser()
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] Checkpoint directory {ckpt_path} not found")

    token = ensure_token(args.token)
    api = HfApi(token=token)

    # Create repo if needed (does nothing if exists and you have write access)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {ckpt_path} → https://huggingface.co/{args.repo_id} …")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(ckpt_path),
        path_in_repo=".",
        repo_type="model",
        token=token,
        commit_message="Checkpoint upload",
        ignore_patterns=["*.pt", "*.log", "*.tmp"],  # tweak as needed
    )
    print("✅ Upload complete.")


if __name__ == "__main__":
    main() 