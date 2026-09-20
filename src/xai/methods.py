"""
XAI (Explainable AI) methods implementation.
Supports: GradCAM, GradCAM++, Integrated Gradients, LIME, SHAP, Occlusion Sensitivity.
"""

import math
import gc
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients
import lime
import lime.lime_image
import shap


def _norm01(m: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    m = np.maximum(m, 0)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def tensor_to_rgb(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """Convert normalized tensor to RGB image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1).astype(np.float32)


class ViTReshapeTransform:
    """Reshape transform for Vision Transformer patch tokens."""
    
    def __init__(self, patch_size: int = 16, img_size: int = 224):
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
    
    def __call__(self, tensor):
        """(B, N, C) -> (B, C, H, W)"""
        n = tensor.size(1)
        s = int(math.isqrt(n))
        if s * s != n:  # e.g., 197 -> drop CLS -> 196
            tensor = tensor[:, n - s * s:, :]
            s = int(math.isqrt(tensor.size(1)))
        return tensor.reshape(tensor.size(0), s, s, tensor.size(2)).permute(0, 3, 1, 2)


def get_gradcam_target_layers(model: nn.Module, model_name: str) -> tuple:
    """Get target layers and reshape transform for GradCAM based on model architecture."""
    
    if model_name == 'resnet50':
        return [model.backbone.layer4[-1].conv3], None
    
    elif model_name == 'densenet121':
        return [model.backbone.features.denseblock4.denselayer16.conv2], None
    
    elif model_name == 'vit_b16':
        # ViT: use last block's norm1
        return [model.backbone.blocks[-1].norm1], ViTReshapeTransform()
    
    elif model_name == 'biomedclip':
        # BiomedCLIP: use last block's norm1 in visual encoder
        return [model.encoder.trunk.blocks[-1].norm1], ViTReshapeTransform()
    
    elif model_name == 'unet':
        return [model.model.decoder.blocks[-1].conv2[0]], None
    
    elif model_name == 'transunet':
        # SegFormer decoder last conv
        return [model.model.decoder.blocks[-1].conv2[0]], None
    
    elif model_name == 'sam_adapter':
        return [model.mask_decoder.output_upscaling[0]], None
    
    return None, None


def compute_gradcam(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    target: Any,
    device: torch.device,
    use_plus_plus: bool = False
) -> np.ndarray:
    """Compute GradCAM or GradCAM++ for a model."""
    
    model.eval()
    target_layers, reshape_transform = get_gradcam_target_layers(model, model_name)
    
    if target_layers is None:
        raise ValueError(f"No target layers defined for {model_name}")
    
    # For transformer models, ensure gradients flow
    if reshape_transform is not None:
        for p in model.parameters():
            p.requires_grad_(True)
    
    CamClass = GradCAMPlusPlus if use_plus_plus else GradCAM
    
    with CamClass(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]
    
    return _norm01(grayscale_cam)


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
    internal_batch_size: int = 4,
    device: torch.device = None
) -> np.ndarray:
    """Compute Integrated Gradients attribution."""
    
    model.eval()
    if device is None:
        device = input_tensor.device
    
    ig = IntegratedGradients(model)
    inp = input_tensor.clone().requires_grad_(True)
    
    attrs = ig.attribute(
        inp,
        torch.zeros_like(inp),
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size
    )
    
    # Sum across channels, normalize
    ig_map = attrs[0].cpu().detach().numpy()
    ig_map = np.abs(ig_map).sum(axis=0)
    ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    
    return ig_map


def compute_lime(
    model: nn.Module,
    rgb_image: np.ndarray,
    target_class: int,
    num_samples: int = 200,
    num_features: int = 10,
    hide_color: int = 0,
    segmenter: str = "quickshift",
    segmenter_params: dict = None,
    device: torch.device = None,
    mean: list = None,
    std: list = None
) -> np.ndarray:
    """Compute LIME explanation."""
    
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    def make_predictor(mdl):
        def predictor(images):
            mdl.eval()
            batch = []
            for img in images:
                t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                t = T.Normalize(mean, std)(t)
                batch.append(t)
            with torch.no_grad():
                out = mdl(torch.stack(batch).to(device))
            return F.softmax(out, dim=1).cpu().numpy()
        return predictor
    
    from torchvision import transforms as T
    
    lime_img = (rgb_image * 255).astype(np.uint8)
    explainer = lime.lime_image.LimeImageExplainer()
    
    if segmenter != "quickshift":
        raise ValueError(f"Unsupported LIME segmenter: {segmenter}")
    from skimage.segmentation import quickshift
    params = segmenter_params or {}
    segmentation_fn = lambda image: quickshift(image, **params)
    explanation = explainer.explain_instance(
        lime_img,
        make_predictor(model),
        labels=(target_class,),
        num_samples=num_samples,
        hide_color=hide_color,
        segmentation_fn=segmentation_fn
    )
    
    _, mask = explanation.get_image_and_mask(
        target_class, positive_only=True, num_features=num_features, hide_rest=False
    )
    
    return mask.astype(np.float32)


def compute_shap_kernel(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    background_data: np.ndarray,
    background_clusters: int = 10,
    nsamples: int = 32,
    device: torch.device = None
) -> np.ndarray:
    """Compute SHAP KernelExplainer attribution."""
    
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # KernelExplainer calls a NumPy predictor. Keep its temporary CPU move
    # contained so subsequent methods receive a model on the requested device.
    original_device = next(model.parameters()).device
    model.cpu()
    
    def predict_fn(x_flat):
        t = torch.tensor(x_flat, dtype=torch.float32)
        t = t.reshape(-1, 3, 224, 224)
        with torch.no_grad():
            out = F.softmax(model(t), dim=1)
        return out.numpy()
    
    # Summarize background
    bg_summ = shap.kmeans(background_data, background_clusters)
    demo_flat = input_tensor.cpu().numpy().reshape(1, -1)
    
    explainer = shap.KernelExplainer(predict_fn, bg_summ)
    sv = explainer.shap_values(demo_flat, nsamples=nsamples, silent=True)
    
    if isinstance(sv, list):
        sv_cls = sv[target_class][0]
    else:
        sv_cls = sv[0, :, target_class]
    
    # Reshape and normalize
    sal = np.abs(np.array(sv_cls)).reshape(3, 224, 224).sum(axis=0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    
    model.to(original_device if device is None else device)
    return sal


def compute_occlusion(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    patch_size: int = 16,
    stride: int = 8,
    device: torch.device = None
) -> np.ndarray:
    """Compute Occlusion Sensitivity map."""
    
    model.eval()
    if device is None:
        device = input_tensor.device
    
    _, C, H, W = input_tensor.shape
    
    with torch.no_grad():
        base_prob = F.softmax(model(input_tensor), dim=1)[0, target_class].item()
    
    heatmap = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = input_tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0
            
            with torch.no_grad():
                p = F.softmax(model(occluded), dim=1)[0, target_class].item()
            
            drop = base_prob - p
            heatmap[y:y+patch_size, x:x+patch_size] += drop
            count[y:y+patch_size, x:x+patch_size] += 1
    
    heatmap = heatmap / (count + 1e-8)
    heatmap = np.clip(heatmap, 0, None)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


def compute_segmentation_gradcam(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    pred_mask: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """Compute GradCAM for segmentation models."""
    
    model.eval()
    target_layers, _ = get_gradcam_target_layers(model, model_name)
    
    if target_layers is None:
        raise ValueError(f"No target layers for segmentation model {model_name}")
    
    targets = [SemanticSegmentationTarget(0, pred_mask)]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        gc_map = cam(input_tensor=input_tensor, targets=targets)[0]
    
    return gc_map


def compute_segmentation_ig(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_steps: int = 30,
    internal_batch_size: int = 2,
    device: torch.device = None
) -> np.ndarray:
    """Compute Integrated Gradients for segmentation (scalar output)."""
    
    if device is None:
        device = input_tensor.device
    
    # Wrap model to output scalar (mean foreground probability)
    class SegScalarWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    
    wrapped = SegScalarWrapper(model).to(device).eval()
    
    ig = IntegratedGradients(wrapped)
    inp = input_tensor.clone().requires_grad_(True)
    
    attrs = ig.attribute(inp, torch.zeros_like(inp), target=0,
                         n_steps=n_steps, internal_batch_size=internal_batch_size)
    
    ig_map = attrs[0].cpu().detach().abs().sum(0).numpy()
    ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    
    return ig_map


def compute_segmentation_occlusion(
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: int = 28,
    stride: int = 14,
    device: torch.device = None
) -> np.ndarray:
    """Compute Occlusion Sensitivity for segmentation."""
    
    if device is None:
        device = input_tensor.device
    
    class SegScalarWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    
    wrapped = SegScalarWrapper(model).to(device).eval()
    
    return compute_occlusion(wrapped, input_tensor, target_class=0,
                             patch_size=patch_size, stride=stride, device=device)


def compute_segmentation_lime(
    model: nn.Module,
    rgb_image: np.ndarray,
    num_samples: int = 100,
    num_features: int = 10,
    hide_color: int = 0,
    device: torch.device = None,
    mean: list = None,
    std: list = None
) -> np.ndarray:
    """Compute LIME for segmentation."""
    
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    class SegScalarWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    
    wrapped = SegScalarWrapper(model).to(device).eval()
    
    def make_predictor(w):
        def predictor(images):
            w.eval()
            batch = []
            for img in images:
                t = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
                t = T.Normalize(mean, std)(t)
                batch.append(t)
            with torch.no_grad():
                out = torch.sigmoid(w(torch.stack(batch).to(device)))
            p = out.cpu().numpy()
            return np.concatenate([1-p, p], axis=1)
        return predictor
    
    from torchvision import transforms as T
    
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb_image, make_predictor(wrapped),
        top_labels=1, num_samples=num_samples, hide_color=hide_color
    )
    
    top_label = explanation.top_labels[0]
    _, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=num_features, hide_rest=False
    )
    
    return mask.astype(np.float32)


def compute_segmentation_shap_kernel(
    model: nn.Module,
    input_tensor: torch.Tensor,
    background_data: np.ndarray,
    nsamples: int = 32,
    device: torch.device = None
) -> np.ndarray:
    """Compute SHAP KernelExplainer for segmentation."""
    
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # Move all to CPU
    model.cpu()
    
    class SegScalarWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return m(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    
    wrapped = SegScalarWrapper(model)
    
    def predict_fn(x_flat):
        t = torch.tensor(x_flat, dtype=torch.float32)
        t = t.reshape(-1, 3, 224, 224)
        with torch.no_grad():
            out = torch.sigmoid(wrapped(t))
        return out.mean(dim=(1, 2, 3)).numpy().reshape(-1, 1)
    
    bg_summ = shap.kmeans(background_data, 4)
    demo_flat = input_tensor.cpu().numpy().reshape(1, -1)
    
    explainer = shap.KernelExplainer(predict_fn, bg_summ)
    sv = explainer.shap_values(demo_flat, nsamples=nsamples, silent=True)
    
    if isinstance(sv, list):
        sv_arr = sv[0][0]
    else:
        sv_arr = sv[0]
    
    sal = np.abs(sv_arr).reshape(3, 224, 224).sum(axis=0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    
    return sal
