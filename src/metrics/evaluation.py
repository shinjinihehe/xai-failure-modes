"""
Evaluation metrics for XAI faithfulness.
Insertion AUC, Deletion AUC, Sparsity, Pointing Game.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, List


def faithfulness_deletion(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    n_steps: int = 10,
    device: torch.device = None
) -> Tuple[float, List[float]]:
    """
    Deletion AUC: progressively remove most salient pixels.
    Lower AUC = more faithful (removing salient pixels drops confidence).
    """
    model.eval()
    if device is None:
        device = img_tensor.device
    
    flat_idx = np.argsort(saliency_map.flatten())[::-1]  # Most salient first
    n_pixels = len(flat_idx)
    img_np = img_tensor[0].cpu().numpy()
    probs = []
    
    for step in range(n_steps + 1):
        k = int(step / n_steps * n_pixels)
        occ_img = img_np.copy().reshape(3, -1)
        if k > 0:
            occ_img[:, flat_idx[:k]] = 0
        t = torch.tensor(occ_img.reshape(1, 3, *img_tensor.shape[2:]),
                         dtype=torch.float32).to(device)
        with torch.no_grad():
            p = F.softmax(model(t), dim=1)[0, target_class].item()
        probs.append(p)
    
    # np.trapezoid was introduced after NumPy 1.24; np.trapz keeps the
    # published environment compatible with the declared dependency floor.
    auc = float(np.trapz(probs, dx=1/n_steps))
    return auc, probs


def faithfulness_insertion(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    n_steps: int = 10,
    device: torch.device = None
) -> Tuple[float, List[float]]:
    """
    Insertion AUC: progressively add most salient pixels to blurred baseline.
    Higher AUC = more faithful (adding salient pixels recovers confidence).
    """
    model.eval()
    if device is None:
        device = img_tensor.device
    
    flat_idx = np.argsort(saliency_map.flatten())[::-1]
    n_pixels = len(flat_idx)
    img_np = img_tensor[0].cpu().numpy()
    
    # Create blurred baseline
    blurred = cv2.GaussianBlur(
        img_np.transpose(1, 2, 0), (31, 31), 0
    ).transpose(2, 0, 1)
    
    probs = []
    for step in range(n_steps + 1):
        k = int(step / n_steps * n_pixels)
        ins_img = blurred.copy().reshape(3, -1)
        if k > 0:
            ins_img[:, flat_idx[:k]] = img_np.reshape(3, -1)[:, flat_idx[:k]]
        t = torch.tensor(ins_img.reshape(1, 3, *img_tensor.shape[2:]),
                         dtype=torch.float32).to(device)
        with torch.no_grad():
            p = F.softmax(model(t), dim=1)[0, target_class].item()
        probs.append(p)
    
    auc = float(np.trapz(probs, dx=1/n_steps))
    return auc, probs


def seg_faithfulness(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    mode: str = 'deletion',
    n_steps: int = 10,
    device: torch.device = None
) -> float:
    """
    Segmentation faithfulness using mean foreground probability.
    """
    model.eval()
    if device is None:
        device = img_tensor.device
    
    idx = np.argsort(saliency_map.flatten())[::-1]
    N = len(idx)
    img = img_tensor[0].cpu().numpy().reshape(3, -1)
    
    base = None
    if mode == 'insertion':
        base = cv2.GaussianBlur(
            img_tensor[0].cpu().numpy().transpose(1, 2, 0),
            (31, 31), 0
        ).transpose(2, 0, 1).reshape(3, -1)
    
    def fg_score(mdl, t):
        with torch.no_grad():
            return torch.sigmoid(mdl(t))[0, 0].mean().item()
    
    scores = []
    for s in range(n_steps + 1):
        k = int(s / n_steps * N)
        cur = img.copy() if mode == 'deletion' else base.copy()
        if k > 0:
            cur[:, idx[:k]] = 0 if mode == 'deletion' else img[:, idx[:k]]
        t = torch.tensor(cur.reshape(1, 3, *img_tensor.shape[2:]),
                         dtype=torch.float32).to(device)
        scores.append(fg_score(model, t))
    
    return float(np.trapz(scores, dx=1 / n_steps))


def pointing_game(
    saliency_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5
) -> int:
    """
    Pointing Game: check if max saliency point falls in ground truth mask.
    Returns 1 (hit) or 0 (miss).
    """
    peak = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    return int(gt_mask[peak] > threshold)


def sparsity(
    saliency_map: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Sparsity: fraction of pixels above threshold.
    Lower = more focused explanation.
    """
    return float((saliency_map > threshold).mean())


def compute_all_metrics_classification(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int,
    n_steps: int = 10,
    sparsity_threshold: float = 0.5,
    device: torch.device = None
) -> dict:
    """Compute all metrics for a classification explanation."""
    
    del_auc, _ = faithfulness_deletion(model, img_tensor, saliency_map, target_class, n_steps, device)
    ins_auc, _ = faithfulness_insertion(model, img_tensor, saliency_map, target_class, n_steps, device)
    spar = sparsity(saliency_map, sparsity_threshold)
    
    return {
        'Del_AUC': round(del_auc, 4),
        'Ins_AUC': round(ins_auc, 4),
        'Sparsity': round(spar, 4),
    }


def compute_all_metrics_segmentation(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    gt_mask: np.ndarray,
    n_steps: int = 10,
    sparsity_threshold: float = 0.5,
    pointing_threshold: float = 0.5,
    device: torch.device = None
) -> dict:
    """Compute all metrics for a segmentation explanation."""
    
    del_auc = seg_faithfulness(model, img_tensor, saliency_map, 'deletion', n_steps, device)
    ins_auc = seg_faithfulness(model, img_tensor, saliency_map, 'insertion', n_steps, device)
    spar = sparsity(saliency_map, sparsity_threshold)
    point = pointing_game(saliency_map, gt_mask, pointing_threshold)
    
    return {
        'Del_AUC': round(del_auc, 4),
        'Ins_AUC': round(ins_auc, 4),
        'Sparsity': round(spar, 4),
        'Pointing': point,
    }


def aggregate_by_family(df, family_map: dict) -> dict:
    """Aggregate metrics by architecture family."""
    import pandas as pd
    
    df = df.copy()
    df['Family'] = df['Model'].map(family_map)
    
    summary = df.groupby(['Family', 'XAI_Tool'])[
        ['Ins_AUC', 'Del_AUC', 'Sparsity']
    ].mean().reset_index()
    
    return summary
