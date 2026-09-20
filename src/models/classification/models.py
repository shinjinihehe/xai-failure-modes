"""
Classification model definitions for CNN, Transformer, and VLM backbones.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
import open_clip
from typing import Optional, Dict, Any


class ResNet50Classifier(nn.Module):
    """ResNet-50 for 3-class classification."""
    
    def __init__(self, num_classes: int = 3, pretrained: str = "imagenet1k_v2"):
        super().__init__()
        weights = getattr(models.ResNet50_Weights, pretrained.upper()) if hasattr(models.ResNet50_Weights, pretrained.upper()) else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 for 3-class classification."""
    
    def __init__(self, num_classes: int = 3, pretrained: str = "imagenet1k_v1"):
        super().__init__()
        weights = getattr(models.DenseNet121_Weights, pretrained.upper()) if hasattr(models.DenseNet121_Weights, pretrained.upper()) else None
        self.backbone = models.densenet121(weights=weights)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class ViTClassifier(nn.Module):
    """Vision Transformer (ViT-B/16) for classification."""
    
    def __init__(self, num_classes: int = 3, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class BiomedCLIPClassifier(nn.Module):
    """BiomedCLIP with frozen encoder and trainable LayerNorm+Linear head."""
    
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        num_classes: int = 3,
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 2
    ):
        super().__init__()
        
        # Load BiomedCLIP
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.encoder = clip_model.visual
        
        # Auto-detect embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.encoder(dummy).shape[-1]
        
        # Freeze encoder by default
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        # Unfreeze last N transformer blocks for fine-tuning
        if unfreeze_last_n_blocks > 0:
            for name, param in self.encoder.named_parameters():
                if any(f'blocks.{i}' in name for i in range(12 - unfreeze_last_n_blocks, 12)):
                    param.requires_grad = True
        
        # Classification head: LayerNorm + Linear
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


def build_classification_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Factory function to build classification models from config."""
    model_type = model_config.get('backbone', '').lower()
    
    if 'resnet50' in model_type:
        model = ResNet50Classifier(
            num_classes=model_config.get('num_classes', 3),
            pretrained=model_config.get('pretrained', 'imagenet1k_v2')
        )
    elif 'densenet121' in model_type:
        model = DenseNet121Classifier(
            num_classes=model_config.get('num_classes', 3),
            pretrained=model_config.get('pretrained', 'imagenet1k_v1')
        )
    elif 'vit' in model_type and 'biomedclip' not in model_type:
        model = ViTClassifier(
            num_classes=model_config.get('num_classes', 3),
            model_name=model_config.get('backbone', 'vit_base_patch16_224'),
            pretrained=model_config.get('pretrained', True)
        )
    elif 'biomedclip' in model_type:
        model = BiomedCLIPClassifier(
            model_name=model_config.get('backbone', 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'),
            num_classes=model_config.get('num_classes', 3),
            freeze_encoder=model_config.get('freeze_encoder', True),
            unfreeze_last_n_blocks=model_config.get('unfreeze_last_n_blocks', 2)
        )
    else:
        raise ValueError(f"Unknown classification model type: {model_type}")
    
    return model.to(device)


def get_model_family(model_name: str) -> str:
    """Get architecture family for a model name."""
    families = {
        'resnet50': 'CNN',
        'densenet121': 'CNN',
        'vit_b16': 'Transformer',
        'biomedclip': 'VLM',
    }
    return families.get(model_name, 'Unknown')
