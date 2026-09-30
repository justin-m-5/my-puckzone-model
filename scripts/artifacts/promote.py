"""
scripts/artifacts/promote.py — make the candidate models live.

1. Archives the current live files to artifacts/archive/<timestamp>/
2. Moves candidates/win_model.pkl and candidates/goals_model.pkl into place
3. Pushes the live artifacts to Supabase Storage (scripts.artifacts.sync)

xg_model.pkl is deliberately not part of this: ingest scores every shot with
its own copy, and swapping it mid-season makes stored xG inconsistent.

Usage:
    PYTHONPATH=. python3 -m scripts.artifacts.promote            # asks to confirm
    PYTHONPATH=. python3 -m scripts.artifacts.promote --yes
    PYTHONPATH=. python3 -m scripts.artifacts.promote --rollback artifacts/archive/<timestamp>
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys

CANDIDATE_DIR = "candidates"
ARCHIVE_DIR = os.path.join("artifacts", "archive")
PROMOTED = ("win_model.pkl", "goals_model.pkl")


def _archive_live() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(ARCHIVE_DIR, stamp)
    os.makedirs(dest, exist_ok=True)
    for name in PROMOTED:
        if os.path.exists(name):
            shutil.copy2(name, os.path.join(dest, name))
    return dest


def _push() -> None:
    from scripts.artifacts.sync import push
    push(".")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Promote candidate models to live.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--rollback", metavar="ARCHIVE", help="Restore live models from an archive folder")
    args = parser.parse_args(argv)

    source = args.rollback or CANDIDATE_DIR
    missing = [n for n in PROMOTED if not os.path.exists(os.path.join(source, n))]
    if missing:
        print(f"ERROR: {source}/ is missing {', '.join(missing)}")
        return 1

    action = f"restore from {source}" if args.rollback else "promote candidates"
    if not args.yes and input(f"About to {action} and push to Supabase Storage. Continue? [y/N]: ").strip().lower() != "y":
        print("Nothing changed.")
        return 0

    archive = _archive_live()
    print(f"Archived current live models to {archive}/")
    for name in PROMOTED:
        shutil.copy2(os.path.join(source, name), name)
        print(f"  live {name} <- {source}/{name}")
    if not args.rollback:
        for name in PROMOTED:
            os.remove(os.path.join(CANDIDATE_DIR, name))

    _push()
    print(f"\nDone. Roll back with: PYTHONPATH=. python3 -m scripts.artifacts.promote --rollback {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
