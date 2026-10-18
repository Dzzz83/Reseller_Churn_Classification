"""Attribution control for the provisional annual product-density ranking signal."""

import argparse
import json
from pathlib import Path

from comparisons.baseline_diagnosis.inputs import require
from comparisons.period_product_ranking import ROOT, run
from reseller_churn.artifacts import verify_manifest


def main(output):
    source = ROOT / "results/period-product-ranking-20261018"
    verify_manifest(source)
    decision = json.loads(
        (source / "period_product_density/ranking_decision.json").read_text()
    )
    require(
        all(decision["checks"].values()),
        "Contingent control requires the density ranking point gates",
    )
    run(
        output,
        arms=[("period_order_exposure", "n_orders_12m", 14)],
        protocol_path=ROOT / "comparisons/period_order_exposure_protocol.md",
        entrypoint=Path(__file__),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
