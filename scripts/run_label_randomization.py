"""Train a shuffled-label ResNet-50 control and retain its attribution check."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from data.datasets import BUSIDataset
from models.classification.models import build_classification_model
from training.trainer import CheckpointManager, train_classifier
from utils.config import get_config
from xai.methods import compute_integrated_gradients

class ShuffledLabels(Dataset):
    def __init__(self, base, seed: int):
        self.base = base
        labels = [base[i][1] for i in range(len(base))]
        self.labels = np.random.default_rng(seed).permutation(labels).tolist()
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        image, _, path = self.base[index]
        return image, self.labels[index], path

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=None); a = p.parse_args()
    cfg = get_config(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, test = BUSIDataset.create_dataloaders(cfg)
    shuffled = DataLoader(ShuffledLabels(train.dataset, cfg.project.seed), batch_size=cfg.training.classification.batch_size, shuffle=True, num_workers=cfg.data.busi.num_workers)
    manager = CheckpointManager(cfg.paths.checkpoints_dir)
    control = build_classification_model(cfg.models.classification.resnet50, device)
    control, _ = train_classifier(control, shuffled, validation, a.epochs or cfg.training.classification.epochs,
        cfg.models.classification.resnet50.lr, cfg.training.classification.weight_decay, manager,
        "resnet50_label_randomized", device, cfg.training.classification.scheduler,
        cfg.training.mixed_precision, cfg.training.gradient_clipping, resume=True)
    reference = build_classification_model(cfg.models.classification.resnet50, device)
    if not manager.load("resnet50", reference, map_location=device): raise FileNotFoundError("resnet50_best.pth is required")
    image, label, path = next(iter(test)); image, label = image[:1].to(device), int(label[0])
    reference_map = compute_integrated_gradients(reference, image, label, cfg.xai.methods.integrated_gradients.n_steps, device=device)
    control_map = compute_integrated_gradients(control, image, label, cfg.xai.methods.integrated_gradients.n_steps, device=device)
    out = Path(cfg.paths.results_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Model": "resnet50", "Image_Path": path[0], "Method": "IntegratedGradients", "Control": "shuffled_label_training", "Spearman": spearmanr(reference_map.ravel(), control_map.ravel()).statistic}]).to_csv(out / "sanity_label_randomization_raw.csv", index=False)

if __name__ == "__main__": main()
