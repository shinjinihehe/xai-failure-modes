"""CNN-only Grad-CAM insertion-AUC stress test for BUSI, ISIC-2018, DRIVE.

The target class is the model prediction because the BUSI classifier is applied
without retraining, exactly as described in the paper's stress-test caveat.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from metrics.evaluation import faithfulness_insertion
from models.classification.models import build_classification_model
from training.trainer import CheckpointManager
from utils.config import get_config
from xai.methods import compute_gradcam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True, choices=("isic2018", "drive")); p.add_argument("--root", required=True); p.add_argument("--limit", type=int, default=20)
    a = p.parse_args(); cfg = get_config(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([Resize((224, 224)), ToTensor(), Normalize(cfg.data.busi.normalization.mean, cfg.data.busi.normalization.std)])
    dataset = ImageFolder(a.root, transform=transform)
    model = build_classification_model(cfg.models.classification.resnet50, device)
    manager = CheckpointManager(cfg.paths.checkpoints_dir)
    if not manager.load("resnet50", model, map_location=device): raise FileNotFoundError("resnet50_best.pth is required")
    rows = []
    for index, (image, _) in enumerate(dataset):
        if index == a.limit: break
        image = image.unsqueeze(0).to(device)
        with torch.no_grad(): target = int(model(image).argmax(1).item())
        saliency = compute_gradcam(model, "resnet50", image, ClassifierOutputTarget(target), device)
        auc, _ = faithfulness_insertion(model, image, saliency, target, cfg.metrics.insertion_deletion.n_steps, device)
        rows.append({"Dataset": a.dataset, "Model": "resnet50", "Method": "GradCAM", "Image_Index": index, "Predicted_Class": target, "Ins_AUC": auc})
    out = Path(cfg.paths.results_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"cross_dataset_{a.dataset}_raw.csv", index=False)

if __name__ == "__main__": main()
