"""Step 5: Aggregate all saved experiment results into a controlled comparison CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile comparison table across experiments.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    tables_dir = Path(args.output_dir) / "tables"
    rows = []

    for json_file in sorted(tables_dir.glob("*.json")):
        if json_file.name == "eda_summary.json":
            continue
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "classification" in data and "calibration" in data:
            cls = data["classification"]
            cal = data["calibration"]
            sel = data["selective"]
            rows.append(
                {
                    "experiment": data.get("experiment", json_file.stem),
                    "accuracy": cls["accuracy"],
                    "balanced_accuracy": cls["balanced_accuracy"],
                    "macro_f1": cls["macro_f1"],
                    "ece_before": cal["ece_uncalibrated"],
                    "ece_after": cal["ece_mc_calibrated"],
                    "brier_after": cal["brier_mc_calibrated"],
                    "aurc": sel["aurc_mc_conf"],
                    "temperature": data.get("learned_temperature", 1.0),
                }
            )
        elif "accuracy" in data and "macro_f1" in data:
            rows.append(
                {
                    "experiment": json_file.stem.replace("_test_metrics", ""),
                    "accuracy": data["accuracy"],
                    "balanced_accuracy": data["balanced_accuracy"],
                    "macro_f1": data["macro_f1"],
                    "ece_before": data.get("ece"),
                    "ece_after": data.get("ece"),
                    "brier_after": data.get("brier_score"),
                    "aurc": data.get("aurc_confidence"),
                    "temperature": 1.0,
                }
            )

    if not rows:
        print(f"No experiment result JSON files found in {tables_dir}.")
        return

    df = pd.DataFrame(rows)
    out_csv = tables_dir / "experiment_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved comparison table to {out_csv}:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
