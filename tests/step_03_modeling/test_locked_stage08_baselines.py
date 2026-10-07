from sklearn.metrics import average_precision_score

from reseller_churn.step_00_config.feature_sets import PRUNED_FEATURES
from reseller_churn.step_00_config.model_settings import (
    MODEL_SEEDS,
    TUNED_GRADIENT_BOOSTING,
    TUNED_RANDOM_FOREST,
)
from reseller_churn.step_00_config.validation_settings import (
    DEVELOPMENT_FOLDS,
)
from reseller_churn.step_01_data.dataset_loader import DatasetLoader
from reseller_churn.step_01_data.temporal_dataset import TemporalDataset
from reseller_churn.step_03_modeling.model_factory import ModelFactory
from reseller_churn.step_03_modeling.resampling import (
    balance_classes_by_random_oversampling,
)
from reseller_churn.step_04_evaluation.regression_checks import (
    RegressionChecks,
)


EXPECTED = {
    "random_forest": {
        "fold_1": 0.409271934480239,
        "fold_2": 0.29468531757418653,
    },
    "gradient_boosting": {
        "fold_1": 0.3480374636919924,
        "fold_2": 0.3057197779457027,
    },
}


def test_locked_stage08_scores() -> None:
    data = DatasetLoader.load_labeled_snapshots()
    temporal_data = TemporalDataset(data)

    for fold in DEVELOPMENT_FOLDS:
        prepared = temporal_data.prepare_fold(fold)

        x_train = prepared.training_data[
            PRUNED_FEATURES
        ]
        y_train = prepared.training_data[
            "churn"
        ].astype(int)

        x_val = prepared.validation_data[
            PRUNED_FEATURES
        ]
        y_val = prepared.validation_data[
            "churn"
        ].astype(int)

        rf_scores = []
        gb_scores = []

        for seed in MODEL_SEEDS:
            rf = ModelFactory.create_random_forest(
                TUNED_RANDOM_FOREST,
                seed,
            )
            rf.fit(x_train, y_train)
            rf_scores.append(
                average_precision_score(
                    y_val,
                    rf.predict_proba(x_val)[:, 1],
                )
            )

            x_balanced, y_balanced = (
                balance_classes_by_random_oversampling(
                    x_train,
                    y_train,
                    seed,
                )
            )

            gb = ModelFactory.create_gradient_boosting(
                TUNED_GRADIENT_BOOSTING,
                seed,
            )
            gb.fit(
                x_balanced,
                y_balanced,
            )
            gb_scores.append(
                average_precision_score(
                    y_val,
                    gb.predict_proba(x_val)[:, 1],
                )
            )

        RegressionChecks.assert_score_matches(
            actual=sum(rf_scores) / len(rf_scores),
            expected=EXPECTED["random_forest"][fold.name],
            label=f"Random Forest {fold.name}",
        )

        RegressionChecks.assert_score_matches(
            actual=sum(gb_scores) / len(gb_scores),
            expected=EXPECTED["gradient_boosting"][fold.name],
            label=f"Gradient Boosting {fold.name}",
        )
