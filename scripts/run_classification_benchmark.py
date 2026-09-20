"""Run the paper's 20-image BUSI classification faithfulness protocol.

Writes raw, one-row-per-image/model/method results before any aggregation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.datasets import BUSIDataset
from metrics.evaluation import compute_all_metrics_classification
from models.classification.models import build_classification_model, get_model_family
from training.trainer import CheckpointManager
from utils.config import get_config
from utils.provenance import write_run_metadata
from xai.methods import (compute_gradcam, compute_integrated_gradients,
                         compute_lime, compute_occlusion, compute_shap_kernel,
                         tensor_to_rgb)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def explain(method, model, name, image, label, rgb, background, cfg, device):
    xai = cfg.xai.methods
    if method == "GradCAM":
        return compute_gradcam(model, name, image, ClassifierOutputTarget(label), device)
    if method == "GradCAM++":
        return compute_gradcam(model, name, image, ClassifierOutputTarget(label), device, True)
    if method == "IntegratedGradients":
        return compute_integrated_gradients(model, image, label, xai.integrated_gradients.n_steps,
                                           xai.integrated_gradients.internal_batch_size, device)
    if method == "LIME":
        return compute_lime(model, rgb, label, xai.lime.num_samples, xai.lime.num_features,
                            xai.lime.hide_color, xai.lime.segmenter,
                            xai.lime.segmenter_params, device,
                            cfg.data.busi.normalization.mean, cfg.data.busi.normalization.std)
    if method == "SHAP":
        return compute_shap_kernel(model, image, label, background,
                                   xai.shap.background_clusters, xai.shap.nsamples, device)
    if method == "Occlusion":
        return compute_occlusion(model, image, label, xai.occlusion.patch_size,
                                 xai.occlusion.stride, device)
    raise ValueError(method)


def main() -> None:
    cfg = get_config()
    seed = cfg.project.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if cfg.project.device == "auto" and torch.cuda.is_available() else "cpu")
    train_loader, _, test_loader = BUSIDataset.create_dataloaders(cfg)
    background = np.concatenate([batch.numpy().reshape(len(batch), -1) for batch, _, _ in train_loader])[:cfg.xai.methods.shap.background_samples]
    subset_size = cfg.xai.evaluation_subset.classification
    samples = []
    for images, labels, paths in test_loader:
        for image, label, path in zip(images, labels, paths):
            samples.append((image, int(label), path))
            if len(samples) == subset_size: break
        if len(samples) == subset_size: break
    if len(samples) != subset_size:
        raise RuntimeError(f"Expected {subset_size} held-out BUSI images; found {len(samples)}")
    checkpoints = CheckpointManager(cfg.paths.checkpoints_dir)
    methods = ("GradCAM", "GradCAM++", "IntegratedGradients", "LIME", "SHAP", "Occlusion")
    rows = []
    for name in ("resnet50", "densenet121", "vit_b16", "biomedclip"):
        model = build_classification_model(cfg.models.classification[name], device)
        if not checkpoints.load(name, model, map_location=device):
            raise FileNotFoundError(f"Missing checkpoint for {name}: {checkpoints.get_checkpoint_path(name)}")
        model.eval()
        for index, (image_cpu, label, path) in enumerate(samples):
            image = image_cpu.unsqueeze(0).to(device)
            rgb = tensor_to_rgb(image_cpu, cfg.data.busi.normalization.mean, cfg.data.busi.normalization.std)
            for method in methods:
                saliency = explain(method, model, name, image, label, rgb, background, cfg, device)
                metrics = compute_all_metrics_classification(model, image, saliency, label,
                    cfg.metrics.insertion_deletion.n_steps, cfg.metrics.sparsity_threshold, device)
                rows.append({"Image_Index": index, "Image_Path": path, "Target_Class": label,
                             "Model": name, "Family": get_model_family(name),
                             "XAI_Tool": method, **metrics})
    results = Path(cfg.paths.results_dir); results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results / "classification_metrics_raw.csv", index=False)
    summary = pd.DataFrame(rows).groupby(["Family", "XAI_Tool"])[["Ins_AUC", "Del_AUC", "Sparsity"]].mean().reset_index()
    summary.to_csv(results / "classification_metrics_family_summary.csv", index=False)
    write_run_metadata(results / "run_metadata.json", ROOT, cfg.to_dict())
    print(f"Saved {len(rows)} raw rows; expected {subset_size * 4 * 6}.")

if __name__ == "__main__":
    main()
