#!/usr/bin/env python
"""Patch Battleship SFT dataset so that <result value="hit"> is turned into
"sunk" or "victory" when the <remaining …/> counts indicate a ship was sunk.

Usage (non-interactive):
    uv python scripts/patch_sink_tags.py \
        --src ljt019/battleship-sft-new-format-patched \
        --dst ljt019/battleship-sft-with-sinks 

Requires: datasets, huggingface_hub (or run `pip install datasets huggingface_hub`).
Make sure you are logged in via `huggingface-cli login` or set HF_TOKEN.
"""

import argparse, re, collections, os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser(description="Patch Battleship dataset sink tags")
    p.add_argument("--repo", default="ljt019/battleship-sft-new-format-patched",
                   help="Dataset repository to patch in-place (HF repo id or local path)")
    p.add_argument("--split", default="train")
    p.add_argument("--no-push", action="store_true",
                   help="Skip pushing back to the hub; store locally only")
    return p.parse_args()


# patterns
_RE_UNKNOWN = re.compile(r'unknown="(\d+)"', re.I)
_RE_REMAIN = re.compile(
    r'<remaining[^>]*?carrier="(\d)"[^>]*?battleship="(\d)"[^>]*?cruiser="(\d)"[^>]*?submarine="(\d)"[^>]*?destroyer="(\d)"',
    re.I,
)
_RE_HIT_BANG = re.compile(r'value=[\"\]?hit![\"\]?', re.I)  # match value="hit!" or value='hit!'


def count_remaining(text: str):
    m = _RE_REMAIN.search(text)
    return sum(map(int, m.groups())) if m else None


def patch_dataset(ds):
    games = collections.defaultdict(list)
    # bucket by game id and sort using unknown count
    for row in ds:
        user_board_msg = row["prompt"][-1]
        unk_match = _RE_UNKNOWN.search(user_board_msg["content"])
        unk_num = int(unk_match.group(1)) if unk_match else -1
        gid = row["answer"]
        games[gid].append((unk_num, row))

    patched_rows = []
    num_sunk_tagged = num_victory_tagged = num_hit_normalized = 0

    for gid, turns in games.items():
        # earliest turn has biggest unknown -> sort desc; then iterate chronologically
        turns.sort(key=lambda t: t[0], reverse=True)
        prev_total = None
        for unk, row in turns:
            # 1) Normalise `hit!` -> `hit` in EVERY message of this row
            for msg in row["prompt"] + row.get("completion", []):
                c = msg.get("content", "")
                new_c, n = _RE_HIT_BANG.subn('value="hit"', c)
                if n:
                    msg["content"] = new_c
                    num_hit_normalized += n

            user_board_msg = row["prompt"][-1]
            content = user_board_msg["content"]
            total = count_remaining(content)
            if total is None:
                prev_total = total
                patched_rows.append(row)
                continue

            if prev_total is not None and total < prev_total:
                # a ship (or more) sunk. last assistant shot is completion of previous turn.
                if total == 0:
                    new_val = "victory"
                    num_victory_tagged += 1
                else:
                    new_val = "sunk"
                    num_sunk_tagged += 1
                # replace hit or hit! (case insensitive)
                content = re.sub(r'value="hit"', f'value="{new_val}"', content, count=1, flags=re.I)
                user_board_msg["content"] = content
            prev_total = total
            patched_rows.append(row)

    print(f"Patched turns: hit!→hit={num_hit_normalized}, sunk={num_sunk_tagged}, victory={num_victory_tagged}")
    return Dataset.from_list(patched_rows)


def main():
    args = parse_args()
    print(f"Loading {args.repo} ({args.split}) …")
    ds = load_dataset(args.repo, split=args.split)
    patched = patch_dataset(ds)
    out_dir = "./patched_dataset_tmp"
    if os.path.exists(out_dir):
        import shutil; shutil.rmtree(out_dir)
    patched.save_to_disk(out_dir)
    print(f"Saved patched dataset to {out_dir} (rows: {len(patched)})")

    if not args.no_push and args.repo.count('/')==1:
        print(f"Pushing patched split back to hub: {args.repo}")
        api = HfApi()

        # STEP 1: delete existing shard & metadata files (separate commit)
        to_delete = [
            f for f in api.list_repo_files(args.repo, repo_type="dataset")
            if f.endswith(".arrow") or f.endswith(".json")
        ]
        if to_delete:
            print(f"Deleting {len(to_delete)} old shard/metadata files …")
            for path in to_delete:
                api.delete_file(
                    path_in_repo=path,
                    repo_id=args.repo,
                    repo_type="dataset",
                    commit_message="remove old dataset shards before sink patch upload",
                )
        else:
            print("No old .arrow/.json files found to delete.")

        # STEP 2: upload new folder (fresh commit)
        api.upload_folder(
            folder_path=out_dir,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="upload patched dataset with sink/victory result tags",
            ignore_patterns=["*.lock"],
        )
        print("Upload complete.")
    else:
        print("Dataset stored locally; not pushed (use without --no-push to upload).")


if __name__ == "__main__":
    main() 