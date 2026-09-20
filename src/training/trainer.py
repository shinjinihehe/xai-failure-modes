"""
Training utilities: checkpoint management, training loops, evaluation.
"""

import os
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score


class CheckpointManager:
    """Manages model checkpoints with auto-resume and Google Drive sync."""
    
    def __init__(self, checkpoint_dir: str, drive_file_ids: Optional[Dict[str, str]] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.drive_file_ids = drive_file_ids or {}
    
    def get_checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}_best.pth"
    
    def exists(self, name: str) -> bool:
        return self.get_checkpoint_path(name).exists()
    
    def load(self, name: str, model: nn.Module, map_location: str = 'cpu') -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        path = self.get_checkpoint_path(name)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=map_location))
            print(f"  Loaded checkpoint: {path.name}")
            return True
        return False
    
    def save(self, name: str, model: nn.Module) -> Path:
        """Save model checkpoint."""
        path = self.get_checkpoint_path(name)
        torch.save(model.state_dict(), path)
        print(f"  Saved checkpoint: {path.name} ({path.stat().st_size/1e6:.1f} MB)")
        return path
    
    def try_download_from_drive(self, name: str) -> bool:
        """Try to download checkpoint from Google Drive."""
        file_id = self.drive_file_ids.get(f"{name}_best.pth", '')
        if not file_id:
            return False
        
        path = self.get_checkpoint_path(name)
        if path.exists():
            return True
        
        try:
            import gdown
            gdown.download(id=file_id, output=str(path), quiet=False)
            print(f"  Downloaded from Drive: {name}")
            return True
        except Exception as e:
            print(f"  Drive download failed for {name}: {e}")
            return False
    
    def auto_resume(self, name: str, model: nn.Module) -> bool:
        """Try to load from disk, then from Drive. Returns True if loaded."""
        if self.load(name, model):
            return True
        # A download alone does not restore the current model.  Reload it so a
        # resumed run never evaluates random initial weights.
        return self.try_download_from_drive(name) and self.load(name, model)
    
    def print_status(self, model_names: list):
        """Print status of all checkpoints."""
        print("=== Checkpoint Status ===")
        for name in model_names:
            path = self.get_checkpoint_path(name)
            if path.exists():
                print(f"  OK  ({path.stat().st_size/1e6:.1f} MB)  {name}")
            else:
                print(f"  --  (not found)  {name}")


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_manager: CheckpointManager,
    model_name: str,
    device: torch.device,
    scheduler_type: str = "cosine",
    mixed_precision: bool = False,
    gradient_clipping: float = 1.0,
    resume: bool = True
) -> Tuple[nn.Module, Dict[str, list]]:
    """Train a classification model with checkpointing."""
    
    # Try to resume from checkpoint
    if resume and checkpoint_manager.auto_resume(model_name, model):
        model.to(device)
        return model, {}
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type == 'cuda' else None
    
    best_val_acc = 0.0
    history = defaultdict(list)
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_correct, train_total = 0, 0
        
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                if gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        
        val_acc = val_correct / val_total
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if scheduler:
            scheduler.step()
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_manager.save(model_name, model)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
    
    print(f"  Best val_acc: {best_val_acc:.4f}")
    
    # Free GPU memory
    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return model, history


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    checkpoint_manager: CheckpointManager,
    model_name: str,
    device: torch.device,
    class_names: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate classifier on test set."""
    checkpoint_manager.load(model_name, model, map_location=device)
    model.eval().to(device)
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    
    print(classification_report(labels, preds, target_names=class_names))
    
    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return preds, labels, probs


def train_segmenter(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_manager: CheckpointManager,
    model_name: str,
    device: torch.device,
    scheduler_type: str = "cosine",
    mixed_precision: bool = False,
    gradient_clipping: float = 1.0,
    resume: bool = True
) -> Tuple[nn.Module, Dict[str, list]]:
    """Train a segmentation model with checkpointing."""
    
    if resume and checkpoint_manager.auto_resume(model_name, model):
        model.to(device)
        return model, {}
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type == 'cuda' else None
    
    best_dice = 0.0
    history = defaultdict(list)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_dices = []
        
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(imgs)
                    loss = criterion(out, masks)
                scaler.scale(loss).backward()
                if gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                loss = criterion(out, masks)
                loss.backward()
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            
            train_dices.append(dice_score(out.detach(), masks).item())
        
        # Validation
        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_dices.append(dice_score(out, masks).item())
        
        td, vd = np.mean(train_dices), np.mean(val_dices)
        history['train_dice'].append(td)
        history['val_dice'].append(vd)
        
        if scheduler:
            scheduler.step()
        
        if vd > best_dice:
            best_dice = vd
            checkpoint_manager.save(model_name, model)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs}  train_dice={td:.4f}  val_dice={vd:.4f}")
    
    print(f"  Best val_dice: {best_dice:.4f}")
    
    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return model, history


def evaluate_segmenter(
    model: nn.Module,
    loader: DataLoader,
    checkpoint_manager: CheckpointManager,
    model_name: str,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate segmenter on test set."""
    checkpoint_manager.load(model_name, model, map_location=device)
    model.eval().to(device)
    
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            dices.append(dice_score(out, masks).item())
            ious.append(iou_score(out, masks).item())
    
    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    print(f"  Test Dice: {mean_dice:.4f}  IoU: {mean_iou:.4f}")
    
    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return mean_dice, mean_iou


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice coefficient for binary segmentation."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU for binary segmentation."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


