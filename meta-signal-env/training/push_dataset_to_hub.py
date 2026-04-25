"""
Push expert_demos.jsonl to a HF Dataset repo.
Run once:  python -m training.push_dataset_to_hub
"""
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

DATASET_REPO = "Anvit2512/meta-signal-expert-demos"
LOCAL_FILE   = Path(__file__).parent.parent / "data" / "expert_demos.jsonl"

if not LOCAL_FILE.exists():
    print(f"ERROR: {LOCAL_FILE} not found. Run generate_dataset.py first.")
    sys.exit(1)

api = HfApi()

print(f"Creating dataset repo: {DATASET_REPO}")
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

print(f"Uploading {LOCAL_FILE.name} ({LOCAL_FILE.stat().st_size/1024/1024:.1f} MB) ...")
api.upload_file(
    path_or_fileobj = str(LOCAL_FILE),
    path_in_repo    = "expert_demos.jsonl",
    repo_id         = DATASET_REPO,
    repo_type       = "dataset",
    commit_message  = "Add Q4 Gauntlet expert demonstrations (10,250 records)",
)
print(f"Done. View at: https://huggingface.co/datasets/{DATASET_REPO}")
