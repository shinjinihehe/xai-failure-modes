"""Recreate aggregate paper figures from immutable raw experiment records."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.visualization import save_family_comparison, save_quantitative_heatmaps

def main() -> None:
    raw = ROOT / "outputs/results/classification_metrics_raw.csv"
    if not raw.exists():
        raise FileNotFoundError("Run run_classification_benchmark.py first")
    df = pd.read_csv(raw)
    expected = {"Image_Index", "Model", "Family", "XAI_Tool", "Ins_AUC", "Del_AUC", "Sparsity"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Raw metrics missing columns: {sorted(expected - set(df.columns))}")
    out = ROOT / "outputs/figures"; out.mkdir(parents=True, exist_ok=True)
    save_quantitative_heatmaps(df, out / "quant_heatmaps.png")
    save_family_comparison(df, out / "family_comparison.png")

if __name__ == "__main__":
    main()
