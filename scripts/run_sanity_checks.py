"""Adebayo-style randomization controls for stored attribution maps.

Runs weight randomization and compares normalized maps using Spearman rank
correlation. Results are retained per image/model for public inspection.
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from data.datasets import BUSIDataset
from models.classification.models import build_classification_model
from training.trainer import CheckpointManager
from utils.config import get_config
from xai.methods import compute_integrated_gradients

def randomize(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "reset_parameters"): module.reset_parameters()

def main() -> None:
    cfg = get_config(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, loader = BUSIDataset.create_dataloaders(cfg)
    image, label, path = next(iter(loader)); image, label = image[:1].to(device), int(label[0])
    manager = CheckpointManager(cfg.paths.checkpoints_dir); rows = []
    for name in ("resnet50", "densenet121", "vit_b16", "biomedclip"):
        model = build_classification_model(cfg.models.classification[name], device)
        if not manager.load(name, model, map_location=device): raise FileNotFoundError(f"Missing checkpoint for {name}")
        original = compute_integrated_gradients(model, image, label, cfg.xai.methods.integrated_gradients.n_steps, device=device)
        randomized = copy.deepcopy(model); randomize(randomized)
        randomized_map = compute_integrated_gradients(randomized, image, label, cfg.xai.methods.integrated_gradients.n_steps, device=device)
        rows.append({"Model": name, "Image_Path": path[0], "Method": "IntegratedGradients", "Control": "full_weight_randomization", "Spearman": spearmanr(original.ravel(), randomized_map.ravel()).statistic})
    out = Path(cfg.paths.results_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sanity_checks_raw.csv", index=False)

if __name__ == "__main__": main()
