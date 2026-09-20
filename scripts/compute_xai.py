"""
XAI computation and evaluation script.
Computes all 6 XAI methods on all 7 models and evaluates faithfulness.
"""

import sys
from pathlib import Path
import torch
import gc
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import get_config
from data.datasets import BUSIDataset, KvasirSEGDataset
from models.classification.models import build_classification_model, get_model_family
from models.segmentation.models import build_segmentation_model, get_segmentation_family
from training.trainer import CheckpointManager
from xai.methods import (
    compute_gradcam, compute_integrated_gradients, compute_lime,
    compute_shap_kernel, compute_occlusion,
    compute_segmentation_gradcam, compute_segmentation_ig,
    compute_segmentation_occlusion, compute_segmentation_lime,
    compute_segmentation_shap_kernel,
    tensor_to_rgb
)
from metrics.evaluation import (
    compute_all_metrics_classification, compute_all_metrics_segmentation
)
from utils.visualization import (
    save_xai_classification_grid, save_xai_segmentation_grid,
    save_quantitative_heatmaps, save_family_comparison,
    save_dataset_samples
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reload_all_models(config, device, ckpt_manager):
    """Load all 7 models from checkpoints."""
    
    models = {}
    
    # Classification models
    clf_models = {}
    for name in ['resnet50', 'densenet121', 'vit_b16', 'biomedclip']:
        model = build_classification_model(config.models.classification[name], device)
        ckpt_manager.load(name, model, map_location=device)
        model.eval().to(device)
        clf_models[name] = model
    
    # Segmentation models
    seg_models = {}
    for name in ['unet', 'transunet']:
        model = build_segmentation_model(config.models.segmentation[name], device)
        ckpt_manager.load(name, model, map_location=device)
        model.eval().to(device)
        seg_models[name] = model
    
    # SAM Adapter (special handling for checkpoint path)
    sam_ckpt = config.models.segmentation.sam_adapter.sam_checkpoint
    sam_config = config.models.segmentation.sam_adapter
    sam_config.sam_checkpoint = sam_ckpt  # Ensure path is set
    model = build_segmentation_model(sam_config, device)
    ckpt_manager.load('sam_adapter', model, map_location=device)
    model.eval().to(device)
    seg_models['sam_adapter'] = model
    
    return clf_models, seg_models


def get_demo_images(config, test_loader, kv_test_loader, device):
    """Get demo images for XAI visualization."""
    
    demo_imgs_clf, demo_labels_clf, _ = next(iter(test_loader))
    demo_imgs_seg, demo_masks_seg, _ = next(iter(kv_test_loader))
    
    demo_img_clf = demo_imgs_clf[0:1].to(device)
    demo_img_seg = demo_imgs_seg[0:1].to(device)
    demo_lbl_clf = int(demo_labels_clf[0])
    
    mean = config.data.busi.normalization.mean
    std = config.data.busi.normalization.std
    
    demo_rgb_clf = tensor_to_rgb(demo_imgs_clf[0], mean, std)
    demo_rgb_seg = tensor_to_rgb(demo_imgs_seg[0], mean, std)
    
    gt_mask_np = demo_masks_seg[0, 0].numpy()
    
    return {
        'demo_img_clf': demo_img_clf,
        'demo_img_seg': demo_img_seg,
        'demo_lbl_clf': demo_lbl_clf,
        'demo_rgb_clf': demo_rgb_clf,
        'demo_rgb_seg': demo_rgb_seg,
        'gt_mask_np': gt_mask_np,
    }


def compute_classification_xai(clf_models, demo_data, config, device):
    """Compute all XAI methods for classification models."""
    
    xai_config = config.xai
    mean = config.data.busi.normalization.mean
    std = config.data.busi.normalization.std
    
    # Background data for SHAP
    train_loader, _, _ = BUSIDataset.create_dataloaders(config)
    bg_list = []
    for imgs, _, _ in train_loader:
        bg_list.append(imgs.numpy().reshape(len(imgs), -1))
        if sum(len(x) for x in bg_list) >= xai_config.methods.shap.background_samples:
            break
    bg_flat = np.concatenate(bg_list)[:xai_config.methods.shap.background_samples]
    
    results = {
        'GradCAM': {}, 'GradCAM++': {}, 'IntGrad': {}, 
        'LIME': {}, 'SHAP': {}, 'Occlusion': {}
    }
    
    for name, model in clf_models.items():
        print(f"\n  Computing XAI for {name}...")
        
        # GradCAM
        if xai_config.methods.gradcam.enabled:
            try:
                results['GradCAM'][name] = compute_gradcam(
                    model, name, demo_data['demo_img_clf'],
                    ClassifierOutputTarget(demo_data['demo_lbl_clf']),
                    device, use_plus_plus=False
                )
            except Exception as e:
                print(f"    GradCAM failed: {e}")
        
        # GradCAM++
        if xai_config.methods.gradcampp.enabled:
            try:
                results['GradCAM++'][name] = compute_gradcam(
                    model, name, demo_data['demo_img_clf'],
                    ClassifierOutputTarget(demo_data['demo_lbl_clf']),
                    device, use_plus_plus=True
                )
            except Exception as e:
                print(f"    GradCAM++ failed: {e}")
        
        # Integrated Gradients
        if xai_config.methods.integrated_gradients.enabled:
            try:
                results['IntGrad'][name] = compute_integrated_gradients(
                    model, demo_data['demo_img_clf'], demo_data['demo_lbl_clf'],
                    n_steps=xai_config.methods.integrated_gradients.n_steps,
                    internal_batch_size=xai_config.methods.integrated_gradients.internal_batch_size,
                    device=device
                )
            except Exception as e:
                print(f"    IntGrad failed: {e}")
        
        # LIME
        if xai_config.methods.lime.enabled:
            try:
                results['LIME'][name] = compute_lime(
                    model, demo_data['demo_rgb_clf'], demo_data['demo_lbl_clf'],
                    num_samples=xai_config.methods.lime.num_samples,
                    num_features=xai_config.methods.lime.num_features,
                    hide_color=xai_config.methods.lime.hide_color,
                    segmenter=xai_config.methods.lime.segmenter,
                    segmenter_params=xai_config.methods.lime.segmenter_params,
                    device=device, mean=mean, std=std
                )
            except Exception as e:
                print(f"    LIME failed: {e}")
        
        # SHAP
        if xai_config.methods.shap.enabled:
            try:
                results['SHAP'][name] = compute_shap_kernel(
                    model, demo_data['demo_img_clf'], demo_data['demo_lbl_clf'],
                    bg_flat,
                    background_clusters=xai_config.methods.shap.background_clusters,
                    nsamples=xai_config.methods.shap.nsamples, device=device
                )
            except Exception as e:
                print(f"    SHAP failed: {e}")
        
        # Occlusion
        if xai_config.methods.occlusion.enabled:
            try:
                results['Occlusion'][name] = compute_occlusion(
                    model, demo_data['demo_img_clf'], demo_data['demo_lbl_clf'],
                    patch_size=xai_config.methods.occlusion.patch_size,
                    stride=xai_config.methods.occlusion.stride,
                    device=device
                )
            except Exception as e:
                print(f"    Occlusion failed: {e}")
        
        # Free memory
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def compute_segmentation_xai(seg_models, demo_data, config, device):
    """Compute all XAI methods for segmentation models."""
    
    xai_config = config.xai
    mean = config.data.busi.normalization.mean
    std = config.data.busi.normalization.std
    
    # Background for SHAP
    _, kv_train_loader, _ = KvasirSEGDataset.create_dataloaders(config)
    seg_bg_list = []
    for imgs, _, _ in kv_train_loader:
        seg_bg_list.append(imgs.numpy().reshape(len(imgs), -1))
        if sum(len(x) for x in seg_bg_list) >= 8:
            break
    seg_bg_flat = np.concatenate(seg_bg_list)[:8]
    
    results = {
        'GradCAM': {}, 'IntGrad': {}, 'Occlusion': {},
        'LIME': {}, 'SHAP': {}
    }
    
    for name, model in seg_models.items():
        print(f"\n  Computing Segmentation XAI for {name}...")
        
        # Get prediction mask for GradCAM
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(demo_data['demo_img_seg']))[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.float32)
        
        # GradCAM
        try:
            results['GradCAM'][name] = compute_segmentation_gradcam(
                model, name, demo_data['demo_img_seg'], pred_mask, device
            )
        except Exception as e:
            print(f"    Seg GradCAM failed: {e}")
        
        # Integrated Gradients
        try:
            results['IntGrad'][name] = compute_segmentation_ig(
                model, demo_data['demo_img_seg'], device=device
            )
        except Exception as e:
            print(f"    Seg IntGrad failed: {e}")
        
        # Occlusion
        try:
            results['Occlusion'][name] = compute_segmentation_occlusion(
                model, demo_data['demo_img_seg'], device=device
            )
        except Exception as e:
            print(f"    Seg Occlusion failed: {e}")
        
        # LIME
        try:
            results['LIME'][name] = compute_segmentation_lime(
                model, demo_data['demo_rgb_seg'],
                num_samples=xai_config.methods.lime.num_samples // 2,  # Fewer samples for seg
                device=device, mean=mean, std=std
            )
        except Exception as e:
            print(f"    Seg LIME failed: {e}")
        
        # SHAP
        try:
            results['SHAP'][name] = compute_segmentation_shap_kernel(
                model, demo_data['demo_img_seg'], seg_bg_flat,
                nsamples=xai_config.methods.shap.nsamples, device=device
            )
        except Exception as e:
            print(f"    Seg SHAP failed: {e}")
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def evaluate_all_xai(clf_models, seg_models, clf_xai, seg_xai, demo_data, config, device):
    """Evaluate all XAI maps with faithfulness metrics."""
    
    n_steps = config.metrics.insertion_deletion.n_steps
    spar_thresh = config.metrics.sparsity_threshold
    point_thresh = config.metrics.pointing_game_threshold
    
    records = []
    
    # Classification metrics
    print("\n=== Evaluating Classification XAI ===")
    for tool_name, tool_maps in clf_xai.items():
        for model_name, model in clf_models.items():
            if model_name not in tool_maps:
                continue
            sal = tool_maps[model_name]
            metrics = compute_all_metrics_classification(
                model, demo_data['demo_img_clf'], sal, demo_data['demo_lbl_clf'],
                n_steps=n_steps, sparsity_threshold=spar_thresh, device=device
            )
            records.append({
                'Task': 'Classification',
                'Family': get_model_family(model_name),
                'Model': model_name,
                'XAI_Tool': tool_name,
                **metrics,
                'Pointing': None
            })
    
    # Segmentation metrics
    print("\n=== Evaluating Segmentation XAI ===")
    for tool_name, tool_maps in seg_xai.items():
        for name, sal in tool_maps.items():
            model = seg_models[name]
            metrics = compute_all_metrics_segmentation(
                model, demo_data['demo_img_seg'], sal, demo_data['gt_mask_np'],
                n_steps=n_steps, sparsity_threshold=spar_thresh,
                pointing_threshold=point_thresh, device=device
            )
            records.append({
                'Task': 'Segmentation',
                'Family': get_segmentation_family(name),
                'Model': name,
                'XAI_Tool': tool_name,
                **metrics
            })
    
    df = pd.DataFrame(records)
    return df


def save_results(df, clf_xai, seg_xai, demo_data, config):
    """Save all results and visualizations."""
    
    results_dir = Path(config.paths.results_dir)
    figures_dir = Path(config.paths.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics CSV
    df.to_csv(results_dir / 'xai_metrics.csv', index=False)
    print(f"\nSaved metrics to {results_dir / 'xai_metrics.csv'}")
    print(df.to_string())
    
    # Save visualizations
    tools_ordered = ['GradCAM', 'GradCAM++', 'IntGrad', 'LIME', 'SHAP', 'Occlusion']
    model_names = ['resnet50', 'densenet121', 'vit_b16', 'biomedclip']
    seg_model_names = ['unet', 'transunet', 'sam_adapter']
    
    # Classification grid
    save_xai_classification_grid(
        None, model_names, clf_xai, tools_ordered,
        demo_data['demo_rgb_clf'],
        figures_dir / 'xai_clf_grid.png',
        figsize=tuple(config.visualization.figsize_clf_grid),
        dpi=config.visualization.dpi
    )
    
    # Segmentation grid
    seg_tool_maps = {
        'GradCAM': seg_xai['GradCAM'],
        'IntGrad': seg_xai['IntGrad'],
        'Occlusion': seg_xai['Occlusion'],
        'LIME': seg_xai['LIME'],
        'SHAP': seg_xai['SHAP']
    }
    save_xai_segmentation_grid(
        seg_model_names, seg_tool_maps, list(seg_tool_maps.keys()),
        demo_data['demo_rgb_seg'], demo_data['gt_mask_np'],
        figures_dir / 'xai_seg_grid.png',
        figsize=tuple(config.visualization.figsize_seg_grid),
        dpi=config.visualization.dpi
    )
    
    # Quantitative heatmaps
    save_quantitative_heatmaps(
        df, figures_dir / 'quant_heatmaps.png',
        figsize=tuple(config.visualization.figsize_heatmaps),
        dpi=config.visualization.dpi
    )
    
    # Family comparison
    save_family_comparison(
        df, figures_dir / 'family_comparison.png',
        figsize=tuple(config.visualization.figsize_family_bars),
        dpi=config.visualization.dpi
    )
    
    print(f"\nAll figures saved to {figures_dir}")


def main():
    config = get_config()
    
    # Set seed
    torch.manual_seed(config.project.seed)
    
    # Device
    if config.project.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.project.device)
    print(f"Device: {device}")
    
    # Checkpoint manager
    drive_file_ids = {
        'resnet50_best.pth':    '1J1bQNEv5l5GQg3mwCU31kqCV3YmfxlSV',
        'densenet121_best.pth': '1XRbZhUd6YSqOxQmHa5BBqNPxpvLGYgU5',
        'vit_b16_best.pth':     '1OgoVueuon9QB7I45kHHTGYxaBS2--e8T',
        'biomedclip_best.pth':  '1YkdeDcXlIsg3bQXECSBNCaoMtDOp9D-H',
        'unet_best.pth':        '1u1dPOvoDpoIVgeCf3juZ2Vaqy0LIGMjm',
        'transunet_best.pth':   '1LcTGCuJ1VJzHj8gQcpzv2B5G00lFYp-l',
        'sam_adapter_best.pth': '1vr5wYBV4pQ0qh3__PByXORL9NKpqzslG',
    }
    ckpt_manager = CheckpointManager(config.paths.checkpoints_dir, drive_file_ids)
    
    # Load data
    _, _, test_loader = BUSIDataset.create_dataloaders(config)
    _, _, kv_test_loader = KvasirSEGDataset.create_dataloaders(config)
    
    # Load all models
    print("\n=== Loading Models ===")
    clf_models, seg_models = reload_all_models(config, device, ckpt_manager)
    print(f"Loaded {len(clf_models)} classification + {len(seg_models)} segmentation models")
    
    # Get demo images
    demo_data = get_demo_images(config, test_loader, kv_test_loader, device)
    print(f"Demo classification label: {config.data.busi.classes[demo_data['demo_lbl_clf']]}")
    
    # Compute XAI
    print("\n=== Computing Classification XAI ===")
    clf_xai = compute_classification_xai(clf_models, demo_data, config, device)
    
    print("\n=== Computing Segmentation XAI ===")
    seg_xai = compute_segmentation_xai(seg_models, demo_data, config, device)
    
    # Evaluate
    df = evaluate_all_xai(clf_models, seg_models, clf_xai, seg_xai, demo_data, config, device)
    
    # Save results
    save_results(df, clf_xai, seg_xai, demo_data, config)
    
    # Coverage check
    print("\n=== Coverage Summary ===")
    expected_clf = set(clf_models.keys())
    expected_seg = set(seg_models.keys())
    
    for tool, tmap in clf_xai.items():
        missing = expected_clf - set(tmap.keys())
        if missing:
            print(f"  CLF {tool:10s} missing: {missing}")
    
    for tool, tmap in seg_xai.items():
        missing = expected_seg - set(tmap.keys())
        if missing:
            print(f"  SEG {tool:10s} missing: {missing}")
    
    print(f"\nTotal records: {len(df)}")
    print("Done!")


if __name__ == "__main__":
    main()
