"""
scripts/validate/artifacts.py — Model artifact policy validator.

Checks that required runtime artifacts exist, reports optional and legacy
artifacts informatively, and exits non-zero only when required artifacts are
missing.

Usage:
    PYTHONPATH=. python3 -m scripts.validate.artifacts
    PYTHONPATH=. python3 -m scripts.validate.artifacts --dir /path/to/models
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Artifact policy
# ---------------------------------------------------------------------------

# Required for v2 core runtime (prediction + serving paths).
REQUIRED_ARTIFACTS: list[str] = [
    "xg_model.pkl",
    "win_model.pkl",
    "goals_model.pkl",
]

# Optional — only needed for player-prop / player-point prediction flows.
OPTIONAL_ARTIFACTS: list[str] = [
    "player_model.pkl",
]

# Legacy — superseded by v2 goals-model path; still referenced by
# scripts/predict/run.py as optional fallbacks but not required for runtime.
# These should live in artifacts/archive/ once formally retired.
LEGACY_ARTIFACTS: list[str] = [
    "score_model.pkl",
    "playoff_model.pkl",
    "playoff_score_model.pkl",
]

# Remediation commands for each required artifact.
_REMEDIATION: dict[str, str] = {
    "xg_model.pkl": "PYTHONPATH=. python3 -m scripts.train.xg",
    "win_model.pkl": "PYTHONPATH=. python3 -m scripts.train.win",
    "goals_model.pkl": "PYTHONPATH=. python3 -m scripts.train.goals",
}

_KNOWN_ARTIFACTS = set(REQUIRED_ARTIFACTS) | set(OPTIONAL_ARTIFACTS) | set(LEGACY_ARTIFACTS)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def validate_artifacts(artifact_dir: str = ".") -> dict:
    """
    Check artifact policy in *artifact_dir*.

    Returns a result dict with keys:
      - missing_required: list[str]  — required files that are absent
      - present_required: list[str]  — required files that are present
      - present_optional: list[str]  — optional files that are present
      - missing_optional: list[str]  — optional files that are absent
      - present_legacy:   list[str]  — legacy files found (warn only)
      - unexpected:       list[str]  — *.pkl files not in any known category
      - passed:           bool        — True iff no required artifacts missing
    """
    present_required: list[str] = []
    missing_required: list[str] = []
    present_optional: list[str] = []
    missing_optional: list[str] = []
    present_legacy: list[str] = []

    for name in REQUIRED_ARTIFACTS:
        if os.path.isfile(os.path.join(artifact_dir, name)):
            present_required.append(name)
        else:
            missing_required.append(name)

    for name in OPTIONAL_ARTIFACTS:
        if os.path.isfile(os.path.join(artifact_dir, name)):
            present_optional.append(name)
        else:
            missing_optional.append(name)

    for name in LEGACY_ARTIFACTS:
        if os.path.isfile(os.path.join(artifact_dir, name)):
            present_legacy.append(name)

    # Detect unexpected *.pkl files (not in any known category).
    try:
        dir_pkls = {f for f in os.listdir(artifact_dir) if f.endswith(".pkl")}
    except OSError:
        dir_pkls = set()
    unexpected = sorted(dir_pkls - _KNOWN_ARTIFACTS)

    return {
        "missing_required": missing_required,
        "present_required": present_required,
        "present_optional": present_optional,
        "missing_optional": missing_optional,
        "present_legacy": present_legacy,
        "unexpected": unexpected,
        "passed": len(missing_required) == 0,
    }


# ---------------------------------------------------------------------------
# CLI reporting
# ---------------------------------------------------------------------------

def _print_report(result: dict, artifact_dir: str) -> None:
    sep = "=" * 60
    print(sep)
    print("  PuckZone model artifact validator")
    print(sep)
    print(f"  Artifact directory: {os.path.abspath(artifact_dir)}")
    print()

    # Required artifacts
    print("── Required artifacts ──────────────────────────────────────")
    for name in REQUIRED_ARTIFACTS:
        if name in result["present_required"]:
            print(f"  ✓  {name}")
        else:
            cmd = _REMEDIATION.get(name, "see README.md for training instructions")
            print(f"  ✗  {name}  [MISSING]")
            print(f"       → Remediate: {cmd}")

    # Optional artifacts
    print()
    print("── Optional artifacts ──────────────────────────────────────")
    for name in OPTIONAL_ARTIFACTS:
        status = "present" if name in result["present_optional"] else "absent (OK)"
        print(f"  ○  {name}  [{status}]")
        if name in result["missing_optional"]:
            print(f"       → Only needed for player-prop flows.")

    # Legacy artifacts
    if result["present_legacy"]:
        print()
        print("── Legacy artifacts (warn only) ────────────────────────────")
        for name in result["present_legacy"]:
            print(f"  ⚠  {name}  [legacy — superseded by v2 goals-model path]")
        print("       → Consider moving these to artifacts/archive/ once no longer needed.")

    # Unexpected artifacts
    if result["unexpected"]:
        print()
        print("── Unexpected *.pkl files (warn only) ──────────────────────")
        for name in result["unexpected"]:
            print(f"  ?  {name}  [not in artifact policy — verify or archive]")

    # Summary
    print()
    print(sep)
    if result["passed"]:
        n = len(result["present_required"])
        print(f"  PASS  — all {n} required artifact(s) present.")
    else:
        missing = ", ".join(result["missing_required"])
        print(f"  FAIL  — missing required artifact(s): {missing}")
        print()
        print("  Remediation steps:")
        for name in result["missing_required"]:
            cmd = _REMEDIATION.get(name, "see README.md")
            print(f"    {cmd}")
    print(sep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate required model artifacts for PuckZone runtime."
    )
    parser.add_argument(
        "--dir",
        default=".",
        metavar="PATH",
        help="Directory to scan for *.pkl artifacts (default: repo root '.').",
    )
    args = parser.parse_args(argv)

    result = validate_artifacts(args.dir)
    _print_report(result, args.dir)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
