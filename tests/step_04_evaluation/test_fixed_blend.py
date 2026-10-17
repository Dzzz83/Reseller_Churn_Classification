"""Fixed blending cannot silently misalign cases, labels, folds or scores."""

import numpy as np
import pandas as pd
import pytest

from comparisons.fixed_blend import COMPONENT, combine_components


def components():
    frame = pd.DataFrame(
        {
            "StoreID": [1, 2],
            "snapshot": pd.to_datetime(["2013-04-01"] * 2),
            "churn": [1, 0],
            "inner_fold": [0, 1],
            "score": [0.2, 0.8],
        }
    )
    return pd.concat(
        [frame.assign(strategy="baseline"), frame.assign(strategy=COMPONENT)],
        ignore_index=True,
    )


def test_fixed_mean_preserves_ties_and_never_reuses_component_cutoff():
    data = components()
    data["cutoff"] = 0.5
    data.loc[data.strategy.eq(COMPONENT), "score"] = [0.8, 0.2]
    result = combine_components(data)
    assert result.score.eq(0.5).all()
    assert "cutoff" not in result


@pytest.mark.parametrize(
    "column,value",
    [("churn", 0), ("inner_fold", 2), ("snapshot", pd.Timestamp("2013-01-01"))],
)
def test_blend_rejects_misaligned_metadata(column, value):
    data = components()
    data.loc[2, column] = value
    with pytest.raises(AssertionError):
        combine_components(data)


def test_blend_rejects_missing_component_and_nonfinite_scores():
    data = components()
    with pytest.raises(ValueError, match="component"):
        combine_components(data[data.strategy.eq("baseline")])
    data.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        combine_components(data)
