"""
Visualization utilities for XAI maps and results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import torch

from pytorch_grad_cam.utils.image import show_cam_on_image


def save_xai_classification_grid(
    models: Dict[str, torch.nn.Module],
    model_names: List[str],
    tool_maps: Dict[str, Dict[str, np.ndarray]],
    tools_ordered: List[str],
    demo_rgb: np.ndarray,
    output_path: Path,
    figsize: tuple = (18, 12),
    dpi: int = 120
):
    """Save classification XAI grid: models x tools."""
    
    map_dicts = [tool_maps[tool] for tool in tools_ordered]
    
    fig, axes = plt.subplots(
        len(model_names) + 1, len(tools_ordered) + 1,
        figsize=figsize
    )
    
    axes[0, 0].axis('off')
    for j, tool in enumerate(tools_ordered):
        axes[0, j+1].text(0.5, 0.5, tool, ha='center', va='center',
                          fontsize=10, fontweight='bold')
        axes[0, j+1].axis('off')
    
    for i, model_name in enumerate(model_names):
        axes[i+1, 0].text(0.5, 0.5, model_name, ha='center', va='center',
                          fontsize=9, fontweight='bold', rotation=90)
        axes[i+1, 0].axis('off')
        for j, (tool, mdict) in enumerate(zip(tools_ordered, map_dicts)):
            ax = axes[i+1, j+1]
            if model_name in mdict:
                overlay = show_cam_on_image(demo_rgb, mdict[model_name], use_rgb=True)
                ax.imshow(overlay)
            else:
                ax.imshow(demo_rgb)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, color='red')
            ax.axis('off')
    
    plt.suptitle('XAI Maps — Classification Models (BUSI)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_xai_segmentation_grid(
    model_names: List[str],
    tool_maps: Dict[str, Dict[str, np.ndarray]],
    tools_ordered: List[str],
    demo_rgb: np.ndarray,
    gt_mask: np.ndarray,
    output_path: Path,
    figsize: tuple = (20, 10),
    dpi: int = 120
):
    """Save segmentation XAI grid: models x tools with GT."""
    
    fig, axes = plt.subplots(
        len(model_names) + 1, len(tools_ordered) + 2,
        figsize=figsize
    )
    
    axes[0, 0].axis('off')
    axes[0, 1].text(0.5, 0.5, 'Image + GT', ha='center', va='center',
                    fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    for j, tool in enumerate(tools_ordered):
        axes[0, j+2].text(0.5, 0.5, tool, ha='center', va='center',
                          fontsize=10, fontweight='bold')
        axes[0, j+2].axis('off')
    
    for i, mname in enumerate(model_names):
        axes[i+1, 0].text(0.5, 0.5, mname, ha='center', va='center',
                          fontsize=9, fontweight='bold', rotation=90)
        axes[i+1, 0].axis('off')
        
        # Image + GT
        ax = axes[i+1, 1]
        ax.imshow(demo_rgb)
        ax.imshow(gt_mask, alpha=0.35, cmap='Reds')
        ax.axis('off')
        
        for j, tool in enumerate(tools_ordered):
            ax = axes[i+1, j+2]
            if mname in tool_maps.get(tool, {}):
                overlay = show_cam_on_image(demo_rgb, tool_maps[tool][mname], use_rgb=True)
                ax.imshow(overlay)
            else:
                ax.imshow(demo_rgb)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, color='red')
            ax.axis('off')
    
    plt.suptitle('XAI Maps — Segmentation Models (Kvasir-SEG)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")



def save_quantitative_heatmaps(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (18, 5),
    dpi: int = 120
):
    """Save quantitative heatmaps: Insertion, Deletion, Sparsity.
    
    Accepts either a combined DataFrame (with Task column) or a
    classification-only DataFrame (as produced by run_classification_benchmark.py).
    """
    
    clf_df = df[df['Task'] == 'Classification'].copy() if 'Task' in df.columns else df.copy()
    
    pivot_ins = clf_df.pivot_table(index='XAI_Tool', columns='Model', values='Ins_AUC')
    pivot_del = clf_df.pivot_table(index='XAI_Tool', columns='Model', values='Del_AUC')
    pivot_sp = clf_df.pivot_table(index='XAI_Tool', columns='Model', values='Sparsity')
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for ax, pivot, title, cmap in zip(
        axes,
        [pivot_ins, pivot_del, pivot_sp],
        ['Insertion AUC (higher=better)', 'Deletion AUC (lower=better)', 'Sparsity (lower=better)'],
        ['YlGn', 'YlOrRd', 'Blues']
    ):
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                    linewidths=0.5, ax=ax, cbar=True)
        ax.set_title(title, fontweight='bold')
    
    plt.suptitle('Quantitative XAI Evaluation — Classification (BUSI)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved: {output_path}")



def save_family_comparison(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (18, 5),
    dpi: int = 120
):
    """Save per-family summary bar chart.
    
    Accepts either a combined DataFrame (with Task column) or a
    classification-only DataFrame (as produced by run_classification_benchmark.py).
    """
    
    clf_df = df[df['Task'] == 'Classification'].copy() if 'Task' in df.columns else df.copy()
    
    if 'Family' not in clf_df.columns:
        raise ValueError("DataFrame must have a 'Family' column. "
                         "Run run_classification_benchmark.py which writes this column.")
    
    family_summary = (
        clf_df.groupby(['Family', 'XAI_Tool'])[['Ins_AUC', 'Del_AUC', 'Sparsity']]
        .mean().reset_index()
    )
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for ax, metric, title in zip(
        axes,
        ['Ins_AUC', 'Del_AUC', 'Sparsity'],
        ['Insertion AUC (higher=better)', 'Deletion AUC (lower=better)', 'Sparsity (lower=better)']
    ):
        pivot = family_summary.pivot(index='XAI_Tool', columns='Family', values=metric)
        pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('XAI Tool')
        ax.legend(title='Architecture Family')
        ax.tick_params(axis='x', rotation=30)
    
    plt.suptitle('XAI Performance by Architecture Family (CNN / Transformer / VLM)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved: {output_path}")



def save_dataset_samples(
    train_loader,
    kv_train_loader,
    busi_classes: List[str],
    output_path: Path,
    figsize: tuple = (14, 6),
    dpi: int = 120
):
    """Save dataset sample visualization."""
    
    inv_norm = torch.nn.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle('Sample Images from Both Datasets', fontsize=14, fontweight='bold')
    
    for i, (img, lbl, _) in enumerate(train_loader):
        if i == 4: break
        ax = axes[0, i]
        ax.imshow(inv_norm(img[0]).permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f'BUSI: {busi_classes[lbl[0]]}', fontsize=9)
        ax.axis('off')
    
    for i, (img, mask, _) in enumerate(kv_train_loader):
        if i == 4: break
        ax = axes[1, i]
        ax.imshow(inv_norm(img[0]).permute(1, 2, 0).clamp(0, 1))
        ax.imshow(mask[0, 0].numpy(), alpha=0.35, cmap='Reds')
        ax.set_title('Kvasir-SEG', fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved: {output_path}")


def save_model_performance(
    clf_results: dict,
    seg_results: dict,
    output_path: Path
):
    """Save model performance summary as CSV."""
    
    clf_df = pd.DataFrame(clf_results).T
    seg_df = pd.DataFrame(seg_results).T
    
    print("\n=== CLASSIFICATION RESULTS ===")
    print(clf_df.to_string())
    print("\n=== SEGMENTATION RESULTS ===")
    print(seg_df.to_string())
    
    clf_df.to_csv(output_path.parent / 'clf_model_results.csv')
    seg_df.to_csv(output_path.parent / 'seg_model_results.csv')
    print(f"\nResults saved to {output_path.parent}")