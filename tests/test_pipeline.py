import subprocess
import sys
from pathlib import Path

import pytest

from reseller_churn.cli import main
from reseller_churn.evaluation import evaluate


def test_standalone_end_to_end_smoke_from_another_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "pipeline/smoke_test.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Synthetic end-to-end smoke passed." in result.stdout


def test_independent_evaluation_requires_approval_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="explicit approval"):
        evaluate(
            tmp_path / "missing.csv",
            tmp_path / "missing-outcomes.csv",
            tmp_path / "report",
            [0.06],
            "2014-04-01",
            "independent",
        )
    assert not (tmp_path / "report").exists()


def test_help_is_available_without_research_inputs(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "evaluate" in capsys.readouterr().out
