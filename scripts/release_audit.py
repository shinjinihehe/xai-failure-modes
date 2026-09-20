"""Fail closed when evidence required for public result claims is absent."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "outputs/splits/busi_seed42.json",
    "outputs/results/classification_metrics_raw.csv",
    "outputs/results/run_metadata.json",
    "outputs/results/model_performance.csv",
    "outputs/results/cross_dataset_isic2018_raw.csv",
    "outputs/results/cross_dataset_drive_raw.csv",
    "outputs/results/sanity_checks_raw.csv",
    "outputs/results/sanity_label_randomization_raw.csv",
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="return non-zero when release evidence is absent")
    args = parser.parse_args()
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    raw = ROOT / "outputs/results/classification_metrics_raw.csv"
    problems = list(missing)
    if raw.is_file():
        lines = raw.read_text().splitlines()
        if len(lines) < 1 + 20 * 4 * 6:
            problems.append("classification_metrics_raw.csv has fewer than 480 result rows")
    if problems:
        print("Release evidence is incomplete:")
        print("\n".join(f"  - {item}" for item in problems))
        return 1 if args.strict else 0
    print("Release evidence package is present.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
