import importlib.util
from pathlib import Path

import pandas as pd
import pytest


# The standalone audit has a CLI-oriented filename; import it for unit tests.
AUDIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "comparisons"
    / "validation_schedule_audit.py"
)
spec = importlib.util.spec_from_file_location("validation_schedule_audit", AUDIT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assess_schedule = module.assess_schedule


def test_actual_history_cannot_support_independent_pretest_holdout():
    bounds = assess_schedule("2011-05-31", "2013-10-01")

    assert bounds.earliest_training == pd.Timestamp("2012-05-31")
    assert bounds.earliest_temporal_validation == pd.Timestamp("2012-11-30")
    assert bounds.temporal_results_available == pd.Timestamp("2013-05-30")
    assert bounds.latest_development_holdout == pd.Timestamp("2013-04-01")
    assert not bounds.has_independent_holdout


def test_earlier_history_makes_a_shared_holdout_possible():
    bounds = assess_schedule("2010-05-31", "2013-10-01")

    assert bounds.has_independent_holdout


def test_holdout_on_day_temporal_labels_become_known_is_allowed():
    bounds = assess_schedule("2011-05-31", "2013-11-30")

    assert bounds.temporal_results_available == bounds.latest_development_holdout
    assert bounds.has_independent_holdout


@pytest.mark.parametrize("history_months,label_months", [(0, 6), (12, 0)])
def test_invalid_window_sizes_are_rejected(history_months, label_months):
    with pytest.raises(ValueError, match="positive"):
        assess_schedule(
            "2011-05-31",
            "2013-10-01",
            history_months=history_months,
            label_months=label_months,
        )
