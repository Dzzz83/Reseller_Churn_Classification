"""Confirmation and simulation-resolution gates shared by frozen development procedures."""

import copy
import json

import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv
from comparisons.bootstrap_resolution import resolution_plan, simulation_audit
from comparisons.matched_evaluation import acceptance, paired_bootstrap
from reseller_churn.artifacts import (
    file_sha256,
    new_output_directory,
    write_json,
    write_manifest,
)


def additional_confirmation(output, candidate_id, tail):
    """Only add gates to a successful primary; never rescue a failed decision."""
    primary_path = output / "decision.json"
    primary = json.loads(primary_path.read_text())
    plan = resolution_plan(tail)
    write_json(output / "resolution_plan.json", plan)
    final = copy.deepcopy(primary)
    stages = {
        "confirmation": "not_run_primary_failed",
        "simulation_resolution": "not_run_primary_failed",
    }
    if primary["accepted"]:
        predictions = read_csv(output / "validation_predictions.csv")
        metrics = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
        confirmation = new_output_directory(output / "confirmation")
        bootstrap = paired_bootstrap(
            predictions, draws=50000, seed=2026, candidate_id=candidate_id
        )
        bootstrap.to_csv(confirmation / "bootstrap.csv", index=False)
        decision, _, _ = acceptance(metrics, bootstrap, tail, candidate_id)
        final["checks"]["fifty_thousand_confirmation"] = decision["accepted"]
        stages["confirmation"] = "passed" if decision["accepted"] else "failed"
        write_json(confirmation / "decision.json", decision)
        if decision["accepted"]:
            print(
                "Primary and 50,000-draw gates pass; running fixed simulation-resolution audit.",
                flush=True,
            )
            audit = simulation_audit(predictions, candidate_id, tail)
            write_json(confirmation / "simulation_resolution.json", audit)
            final["checks"]["simulation_resolution"] = audit["passed"]
            stages["simulation_resolution"] = "passed" if audit["passed"] else "failed"
        else:
            stages["simulation_resolution"] = "not_run_confirmation_failed"
        write_manifest(
            confirmation, dict(stage="additional_uncertainty_gates", models_fitted=0)
        )
    final["accepted"] = all(final["checks"].values())
    write_json(
        output / "final_decision.json",
        dict(
            primary_decision_sha256=file_sha256(primary_path),
            decision=final,
            stages=stages,
            bootstrap_draw_plan=plan,
        ),
    )
    return final
