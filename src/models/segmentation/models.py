"""
Segmentation model definitions for CNN, Transformer, and VLM backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry
from typing import Dict, Any


class UNetSegmenter(nn.Module):
    """U-Net with ResNet-34 encoder for segmentation."""
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    
    def forward(self, x):
        return self.model(x)


class TransUNetSegmenter(nn.Module):
    """SegFormer (MiT-B2) for transformer-based segmentation."""
    
    def __init__(
        self,
        encoder_name: str = "mit_b2",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1
    ):
        super().__init__()
        # Verify SegFormer is available
        if not hasattr(smp, 'Segformer'):
            raise ImportError(
                "SegmentationModelsPytorch doesn't have Segformer. "
                "Please upgrade: pip install -U segmentation-models-pytorch"
            )
        
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    
    def forward(self, x):
        return self.model(x)


class SAMSegAdapter(nn.Module):
    """SAM Adapter - freeze image/prompt encoder, fine-tune mask decoder only."""
    
    def __init__(
        self,
        sam_checkpoint: str,
        image_size: int = 224
    ):
        super().__init__()
        
        # Load base SAM model
        sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        
        # Remove positional embedding (we'll use fixed size)
        sam.image_encoder.pos_embed = None
        
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.image_size = image_size
        
        # Freeze image and prompt encoders
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        
        # Mask decoder stays trainable
        for p in self.mask_decoder.parameters():
            p.requires_grad = True
    
    def forward(self, x):
        B = x.shape[0]
        results = []
        
        for i in range(B):
            xi = x[i:i+1]
            img_emb = self.image_encoder(xi)
            _, C, H, W = img_emb.shape
            
            # No prompts - use empty prompt embeddings
            sparse_emb, dense_emb = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            
            # Resize dense embeddings to match image embedding size
            dense_emb = F.interpolate(dense_emb, size=(H, W),
                                      mode='bilinear', align_corners=False)
            image_pe = self.prompt_encoder.get_dense_pe()
            image_pe = F.interpolate(image_pe, size=(H, W),
                                     mode='bilinear', align_corners=False)
            
            # Predict masks
            low_res_mask, _ = self.mask_decoder(
                image_embeddings=img_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False
            )
            
            # Upsample to original image size
            mask = F.interpolate(low_res_mask, size=(self.image_size, self.image_size),
                                 mode='bilinear', align_corners=False)
            results.append(mask)
        
        return torch.cat(results, dim=0)


def build_segmentation_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Factory function to build segmentation models from config."""
    arch = model_config.get('architecture', '').lower()
    
    if arch == 'unet':
        model = UNetSegmenter(
            encoder_name=model_config.get('encoder', 'resnet34'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet'),
            in_channels=model_config.get('in_channels', 3),
            classes=model_config.get('classes', 1)
        )
    elif arch == 'segformer':
        model = TransUNetSegmenter(
            encoder_name=model_config.get('encoder', 'mit_b2'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet'),
            in_channels=model_config.get('in_channels', 3),
            classes=model_config.get('classes', 1)
        )
    elif arch == 'sam_adapter':
        sam_ckpt = model_config.get('sam_checkpoint', 'sam_vit_b_01ec64.pth')
        # Resolve checkpoint path
        from pathlib import Path
        ckpt_path = Path(sam_ckpt)
        if not ckpt_path.is_absolute():
            # Look in common locations
            for base in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working')]:
                candidate = base / sam_ckpt
                if candidate.exists():
                    ckpt_path = candidate
                    break
        
        model = SAMSegAdapter(
            sam_checkpoint=str(ckpt_path),
            image_size=model_config.get('img_size', 224)
        )
    else:
        raise ValueError(f"Unknown segmentation architecture: {arch}")
    
    return model.to(device)


def get_segmentation_family(model_name: str) -> str:
    """Get architecture family for a segmentation model name."""
    families = {
        'unet': 'CNN',
        'transunet': 'Transformer',
        'sam_adapter': 'VLM',
    }
    return families.get(model_name, 'Unknown')