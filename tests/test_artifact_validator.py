# tests/test_artifact_validator.py
"""
Offline tests for scripts/validate/artifacts.py.

All tests create temporary directories with controlled *.pkl files so no
real model artifacts (and no Supabase / network access) are required.
"""

import os
import pytest

from scripts.validate.artifacts import (
    LEGACY_ARTIFACTS,
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    validate_artifacts,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(directory: str, *names: str) -> None:
    """Create empty files with the given names inside *directory*."""
    for name in names:
        path = os.path.join(directory, name)
        with open(path, "wb"):
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateArtifacts:
    def test_pass_when_all_required_present(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS)
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is True
        assert result["missing_required"] == []
        assert sorted(result["present_required"]) == sorted(REQUIRED_ARTIFACTS)

    def test_fail_when_one_required_missing(self, tmp_path):
        present = REQUIRED_ARTIFACTS[1:]      # drop the first required artifact
        _touch(str(tmp_path), *present)
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is False
        assert REQUIRED_ARTIFACTS[0] in result["missing_required"]
        assert set(present).issubset(set(result["present_required"]))

    def test_fail_when_all_required_missing(self, tmp_path):
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is False
        assert sorted(result["missing_required"]) == sorted(REQUIRED_ARTIFACTS)

    def test_optional_missing_does_not_fail(self, tmp_path):
        # Only required artifacts present, no optional ones.
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS)
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is True
        assert sorted(result["missing_optional"]) == sorted(OPTIONAL_ARTIFACTS)
        assert result["present_optional"] == []

    def test_optional_present_reported(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS)
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is True
        assert sorted(result["present_optional"]) == sorted(OPTIONAL_ARTIFACTS)
        assert result["missing_optional"] == []

    def test_legacy_artifacts_produce_warnings_only(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS, *LEGACY_ARTIFACTS)
        result = validate_artifacts(str(tmp_path))
        # Must still pass even with legacy files present.
        assert result["passed"] is True
        assert sorted(result["present_legacy"]) == sorted(LEGACY_ARTIFACTS)

    def test_legacy_artifacts_absent_is_fine(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS)
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is True
        assert result["present_legacy"] == []

    def test_unexpected_pkl_reported_as_warning(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS, "mystery_model.pkl")
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is True
        assert "mystery_model.pkl" in result["unexpected"]

    def test_no_unexpected_when_only_known_files(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS, *LEGACY_ARTIFACTS)
        result = validate_artifacts(str(tmp_path))
        assert result["unexpected"] == []

    def test_non_pkl_files_ignored(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS)
        # Add non-pkl files; they should not appear in unexpected.
        _touch(str(tmp_path), "README.md", "model_notes.txt", "data.csv")
        result = validate_artifacts(str(tmp_path))
        assert result["passed"] is True
        assert result["unexpected"] == []


class TestMainExitCodes:
    def test_main_returns_0_when_all_required_present(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS)
        exit_code = main(["--dir", str(tmp_path)])
        assert exit_code == 0

    def test_main_returns_1_when_required_missing(self, tmp_path):
        # Provide only optional artifacts — required ones are absent.
        _touch(str(tmp_path), *OPTIONAL_ARTIFACTS)
        exit_code = main(["--dir", str(tmp_path)])
        assert exit_code == 1

    def test_main_returns_0_with_legacy_files_present(self, tmp_path):
        _touch(str(tmp_path), *REQUIRED_ARTIFACTS, *LEGACY_ARTIFACTS)
        exit_code = main(["--dir", str(tmp_path)])
        assert exit_code == 0
