"""One contingent, controlled development test of personalized purchase timing."""

import argparse
import json
from pathlib import Path

from comparisons.baseline_diagnosis.inputs import require
from comparisons.period_product_ranking import ROOT, run
from reseller_churn.artifacts import verify_manifest


def main(output):
    previous = ROOT / "results/period-product-ranking-20261018"
    verify_manifest(previous)
    for name in ["period_product_breadth", "period_product_density"]:
        require(
            not json.loads((previous / name / "decision.json").read_text())["accepted"],
            "Contingent follow-up requires both product trials to fail",
        )
    run(
        output,
        arms=[("period_personalized_recency", "overdue_ratio", 13)],
        protocol_path=ROOT / "comparisons/period_personalized_recency_protocol.md",
        entrypoint=Path(__file__),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
