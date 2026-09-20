"""
Main training script for all 7 models.
"""

import sys
from pathlib import Path
import torch
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import get_config
from data.datasets import BUSIDataset, KvasirSEGDataset
from models.classification.models import build_classification_model, get_model_family
from models.segmentation.models import build_segmentation_model, get_segmentation_family
from training.trainer import (
    CheckpointManager, train_classifier, evaluate_classifier,
    train_segmenter, evaluate_segmenter
)


def main():
    # Load configuration
    config = get_config()
    
    # Set seed
    torch.manual_seed(config.project.seed)
    
    # Device
    if config.project.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.project.device)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Drive file IDs for checkpoint sync
    drive_file_ids = {
        'resnet50_best.pth':    '1J1bQNEv5l5GQg3mwCU31kqCV3YmfxlSV',
        'densenet121_best.pth': '1XRbZhUd6YSqOxQmHa5BBqNPxpvLGYgU5',
        'vit_b16_best.pth':     '1OgoVueuon9QB7I45kHHTGYxaBS2--e8T',
        'biomedclip_best.pth':  '1YkdeDcXlIsg3bQXECSBNCaoMtDOp9D-H',
        'unet_best.pth':        '1u1dPOvoDpoIVgeCf3juZ2Vaqy0LIGMjm',
        'transunet_best.pth':   '1LcTGCuJ1VJzHj8gQcpzv2B5G00lFYp-l',
        'sam_adapter_best.pth': '1vr5wYBV4pQ0qh3__PByXORL9NKpqzslG',
    }
    
    # Checkpoint manager
    ckpt_manager = CheckpointManager(config.paths.checkpoints_dir, drive_file_ids)
    
    # ============================================================
    # DATA LOADERS
    # ============================================================
    print("\n=== Loading Datasets ===")
    train_loader, val_loader, test_loader = BUSIDataset.create_dataloaders(config)
    kv_train_loader, kv_val_loader, kv_test_loader = KvasirSEGDataset.create_dataloaders(config)
    
    # SAM uses smaller batch size
    sam_train_loader, sam_val_loader, sam_test_loader = KvasirSEGDataset.create_sam_dataloaders(
        config, batch_size=config.models.segmentation.sam_adapter.batch_size
    )
    
    # ============================================================
    # CLASSIFICATION MODELS
    # ============================================================
    print("\n=== Training Classification Models ===")
    clf_models = {}
    clf_results = {}
    
    for model_name in ['resnet50', 'densenet121', 'vit_b16', 'biomedclip']:
        model_config = config.models.classification[model_name]
        
        print(f"\n--- {model_name} ---")
        model = build_classification_model(model_config, device)
        
        lr = model_config.lr * model_config.get('lr_multiplier', 1.0)
        
        model, history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.classification.epochs,
            lr=lr,
            weight_decay=config.training.classification.weight_decay,
            checkpoint_manager=ckpt_manager,
            model_name=model_name,
            device=device,
            scheduler_type=config.training.classification.scheduler,
            mixed_precision=config.training.mixed_precision,
            gradient_clipping=config.training.gradient_clipping,
            resume=config.training.classification.resume
        )
        
        clf_models[model_name] = model
        
        # Evaluate
        preds, labels, probs = evaluate_classifier(
            model, test_loader, ckpt_manager, model_name, device,
            class_names=config.data.busi.classes
        )
        
        clf_results[model_name] = {
            'Accuracy': round(accuracy_score(labels, preds), 4),
            'F1-macro': round(f1_score(labels, preds, average='macro'), 4),
        }
    
    # ============================================================
    # SEGMENTATION MODELS
    # ============================================================
    print("\n=== Training Segmentation Models ===")
    seg_models = {}
    seg_results = {}
    
    # U-Net
    print("\n--- unet ---")
    model = build_segmentation_model(config.models.segmentation.unet, device)
    model, history = train_segmenter(
        model=model,
        train_loader=kv_train_loader,
        val_loader=kv_val_loader,
        epochs=config.training.segmentation.epochs,
        lr=config.models.segmentation.unet.lr,
        weight_decay=config.training.segmentation.weight_decay,
        checkpoint_manager=ckpt_manager,
        model_name='unet',
        device=device,
        scheduler_type=config.training.segmentation.scheduler,
        mixed_precision=config.training.mixed_precision,
        gradient_clipping=config.training.gradient_clipping,
        resume=config.training.segmentation.resume
    )
    seg_models['unet'] = model
    dice, iou = evaluate_segmenter(model, kv_test_loader, ckpt_manager, 'unet', device)
    seg_results['unet'] = {'Dice': round(dice, 4), 'IoU': round(iou, 4)}
    
    # TransUNet (SegFormer)
    print("\n--- transunet ---")
    model = build_segmentation_model(config.models.segmentation.transunet, device)
    model, history = train_segmenter(
        model=model,
        train_loader=kv_train_loader,
        val_loader=kv_val_loader,
        epochs=config.training.segmentation.epochs,
        lr=config.models.segmentation.transunet.lr,
        weight_decay=config.training.segmentation.weight_decay,
        checkpoint_manager=ckpt_manager,
        model_name='transunet',
        device=device,
        scheduler_type=config.training.segmentation.scheduler,
        mixed_precision=config.training.mixed_precision,
        gradient_clipping=config.training.gradient_clipping,
        resume=config.training.segmentation.resume
    )
    seg_models['transunet'] = model
    dice, iou = evaluate_segmenter(model, kv_test_loader, ckpt_manager, 'transunet', device)
    seg_results['transunet'] = {'Dice': round(dice, 4), 'IoU': round(iou, 4)}
    
    # SAM Adapter
    print("\n--- sam_adapter ---")
    model = build_segmentation_model(config.models.segmentation.sam_adapter, device)
    model, history = train_segmenter(
        model=model,
        train_loader=sam_train_loader,
        val_loader=sam_val_loader,
        epochs=config.models.segmentation.sam_adapter.epochs,
        lr=config.models.segmentation.sam_adapter.lr * config.models.segmentation.sam_adapter.lr_multiplier,
        weight_decay=config.training.segmentation.weight_decay,
        checkpoint_manager=ckpt_manager,
        model_name='sam_adapter',
        device=device,
        scheduler_type=config.training.segmentation.scheduler,
        mixed_precision=config.training.mixed_precision,
        gradient_clipping=config.training.gradient_clipping,
        resume=config.training.segmentation.resume
    )
    seg_models['sam_adapter'] = model
    dice, iou = evaluate_segmenter(model, sam_test_loader, ckpt_manager, 'sam_adapter', device)
    seg_results['sam_adapter'] = {'Dice': round(dice, 4), 'IoU': round(iou, 4)}
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    import pandas as pd
    from utils.visualization import save_model_performance
    
    print("\n=== Saving Results ===")
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    save_model_performance(clf_results, seg_results, results_dir / 'model_performance.csv')
    
    # Print summary
    print("\n=== FINAL CHECKPOINT STATUS ===")
    ckpt_manager.print_status([
        'resnet50', 'densenet121', 'vit_b16', 'biomedclip',
        'unet', 'transunet', 'sam_adapter'
    ])
    
    print("\nTraining complete!")


if __name__ == "__main__":
    # Need to import these for the last part
    from sklearn.metrics import accuracy_score, f1_score
    main()