from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalFold:
    """One past-to-future validation fold."""

    name: str
    training_snapshots: tuple[str, ...]
    validation_snapshot: str


DEVELOPMENT_FOLDS = (
    TemporalFold(
        name="fold_1",
        training_snapshots=("2012-07-01",),
        validation_snapshot="2013-01-01",
    ),
    TemporalFold(
        name="fold_2",
        training_snapshots=("2012-07-01", "2012-10-01"),
        validation_snapshot="2013-04-01",
    ),
)

LABELED_SNAPSHOTS = (
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
)

ALL_FEATURE_SNAPSHOTS = (
    "2012-07-01",
    "2012-10-01",
    "2013-01-01",
    "2013-04-01",
    "2013-07-01",
    "2013-10-01",
)

FINAL_TEST_SNAPSHOT = "2013-10-01"

EXPECTED_LABELED_COUNTS = {
    "2012-07-01": (326, 76),
    "2012-10-01": (366, 47),
    "2013-01-01": (343, 25),
    "2013-04-01": (340, 64),
}

EXPECTED_FINAL_TEST_ROWS = 492
