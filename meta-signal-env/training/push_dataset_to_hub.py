"""
Push expert_demos.jsonl to a HF Dataset repo.
Run: python -m training.push_dataset_to_hub --token hf_YOURTOKEN
"""
import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DATASET_REPO = "Anvit25/meta-signal-expert-demos"
LOCAL_FILE   = Path(__file__).parent.parent / "data" / "expert_demos.jsonl"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="HF write token (hf_...)")
    args = p.parse_args()

    if not LOCAL_FILE.exists():
        print(f"ERROR: {LOCAL_FILE} not found.")
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"Creating repo: huggingface.co/datasets/{DATASET_REPO}")
    create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, token=args.token)

    size_mb = LOCAL_FILE.stat().st_size / 1024 / 1024
    print(f"Uploading {LOCAL_FILE.name} ({size_mb:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj = str(LOCAL_FILE),
        path_in_repo    = "expert_demos.jsonl",
        repo_id         = DATASET_REPO,
        repo_type       = "dataset",
        commit_message  = "Add Q4 Gauntlet expert demonstrations (10,250 records)",
    )
    print(f"Done: https://huggingface.co/datasets/{DATASET_REPO}")

if __name__ == "__main__":
    main()
