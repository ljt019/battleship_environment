#!/usr/bin/env python3
"""Upload a Battleship-GRPO checkpoint directory to the Hugging Face Hub.

Just edit the constants below to configure the upload.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder, upload_folder

# ===== CONFIGURATION =====
# Local checkpoint folder (must contain config.json, model*.bin, tokenizer files, etc)
CKPT_DIR = "outputs/battleship-grpo-1.7B/checkpoint-1100"

# Destination repo id, e.g. ljt019/Qwen3-1.7B-Battleship-GRPO
REPO_ID = "ljt019/Qwen3-1.7B-battleship-grpo"

# Set to True to create the repo as private
PRIVATE = False

# Optional: Set your HF token here, or use HF_TOKEN env var or cached login
HF_TOKEN = None
# ========================

def ensure_token(token: Optional[str] = None) -> str:
    token = token or os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        sys.exit("[ERROR] No HuggingFace token provided. Set HF_TOKEN env var or run `huggingface-cli login`.")
    return token

def main() -> None:
    ckpt_path = Path(CKPT_DIR).expanduser()
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] Checkpoint directory {ckpt_path} not found")

    token = ensure_token(HF_TOKEN)
    api = HfApi(token=token)

    # Create repo if needed (does nothing if exists and you have write access)
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True,
    )

    print(f"Uploading {ckpt_path} → https://huggingface.co/{REPO_ID} …")
    upload_folder(
        repo_id=REPO_ID,
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