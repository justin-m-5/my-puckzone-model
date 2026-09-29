"""
scripts/artifacts/sync.py — Sync required model artifacts with Supabase Storage.

The .pkl files are gitignored (the repo is public), so scheduled jobs pull them
from a private Supabase Storage bucket instead. Push after every retrain.

Usage:
    PYTHONPATH=. python3 -m scripts.artifacts.sync push   # local -> Storage
    PYTHONPATH=. python3 -m scripts.artifacts.sync pull   # Storage -> local

Requires SUPABASE_SERVICE_ROLE_KEY (the bucket is private).
"""

import argparse
import os
import sys

from db import supabase
from scripts.validate.artifacts import REQUIRED_ARTIFACTS

BUCKET = os.getenv("MODEL_BUCKET", "models")


def ensure_bucket() -> None:
    names = {b.name for b in supabase.storage.list_buckets()}
    if BUCKET not in names:
        supabase.storage.create_bucket(BUCKET, options={"public": False})
        print(f"Created private bucket '{BUCKET}'")


def push(directory: str) -> None:
    ensure_bucket()
    store = supabase.storage.from_(BUCKET)
    for name in REQUIRED_ARTIFACTS:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)
        with open(path, "rb") as f:
            store.upload(name, f.read(), {"content-type": "application/octet-stream", "upsert": "true"})
        print(f"  pushed {name} ({os.path.getsize(path):,} bytes)")


def pull(directory: str) -> None:
    store = supabase.storage.from_(BUCKET)
    for name in REQUIRED_ARTIFACTS:
        data = store.download(name)
        with open(os.path.join(directory, name), "wb") as f:
            f.write(data)
        print(f"  pulled {name} ({len(data):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync model artifacts with Supabase Storage.")
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--dir", default=".", help="Local artifact directory (default: repo root)")
    args = parser.parse_args()
    push(args.dir) if args.action == "push" else pull(args.dir)


if __name__ == "__main__":
    main()
