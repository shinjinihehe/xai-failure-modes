import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "src"))
from metrics.evaluation import faithfulness_deletion, faithfulness_insertion
from utils.config import get_config
from xai.methods import ViTReshapeTransform


class Toy(torch.nn.Module):
    def forward(self, x):
        return torch.stack((x.mean((1,2,3)), -x.mean((1,2,3))), 1)


def test_camera_ready_protocol():
    """Verify the config exactly matches the camera-ready training protocol."""
    cfg = get_config()
    assert cfg.training.classification.epochs == 50, \
        "Paper §III: classification trains for 50 epochs"
    assert cfg.training.segmentation.epochs == 100, \
        "Paper §III: segmentation trains for 100 epochs"
    assert cfg.xai.evaluation_subset.classification == 20, \
        "Paper §III: faithfulness evaluated over 20 held-out images"


def test_xai_hyperparameters():
    """Verify XAI hyperparameters exactly match the camera-ready paper (§III)."""
    cfg = get_config()
    lime = cfg.xai.methods.lime
    assert lime.num_samples == 200, "LIME: num_samples=200"
    assert lime.num_features == 10, "LIME: num_features=10"
    assert lime.hide_color == 0, "LIME: hide_color=0"
    assert lime.segmenter == "quickshift", "LIME: quickshift segmenter"
    shap = cfg.xai.methods.shap
    assert shap.background_clusters == 10, "SHAP: k-means background k=10"
    assert shap.nsamples == 32, "SHAP: 32 eval samples/image"
    occ = cfg.xai.methods.occlusion
    assert occ.patch_size == 16, "Occlusion: 16x16 patch"
    assert occ.stride == 8, "Occlusion: stride 8"
    ig = cfg.xai.methods.integrated_gradients
    assert ig.n_steps == 50, "IG: straight-line path with 50 steps"


def test_training_hyperparameters():
    """Verify training hyperparameters match §III implementation details."""
    cfg = get_config()
    clf = cfg.training.classification
    assert clf.batch_size == 16, "Paper §III: classification batch size 16"
    assert clf.scheduler == "cosine", "Paper §III: cosine LR schedule"
    assert clf.weight_decay == 1e-4, "Paper §III: AdamW weight decay 1e-4"
    for name in ("resnet50", "densenet121", "vit_b16", "biomedclip"):
        lr = cfg.models.classification[name].lr
        assert lr == 3e-5, f"Paper §III: peak LR 3e-5 for {name}"
    seg = cfg.training.segmentation
    assert seg.batch_size == 8, "Paper §III: segmentation batch size 8"
    for seg_name in ("unet", "transunet"):
        lr = cfg.models.segmentation[seg_name].lr
        assert lr == 1e-4, f"Paper §III: peak LR 1e-4 for {seg_name}"


def test_data_split():
    """Verify 70/15/15 stratified split configuration."""
    cfg = get_config()
    busi = cfg.data.busi
    assert busi.train_split == 0.7
    assert busi.val_split == 0.15
    assert busi.test_split == 0.15


def test_faithfulness_metrics_return_finite_auc():
    """Verify insertion/deletion AUC returns finite values on a toy model."""
    image = torch.ones(1, 3, 8, 8)
    saliency = np.ones((8, 8), dtype=np.float32)
    assert np.isfinite(faithfulness_deletion(Toy(), image, saliency, 0, 4)[0])
    assert np.isfinite(faithfulness_insertion(Toy(), image, saliency, 0, 4)[0])


def test_vit_reshape_transform_shape():
    """Verify ViTReshapeTransform produces correct spatial grid from patch tokens.
    
    This is the key approximation used for Grad-CAM on ViT-B/16 and BiomedCLIP.
    ViT-B/16 with 224x224 input → 196 patch tokens + 1 CLS = 197 tokens.
    After dropping CLS: 196 = 14×14 spatial grid.
    """
    transform = ViTReshapeTransform(patch_size=16, img_size=224)
    # Simulate ViT-B/16 output: batch=1, 197 tokens (196 patches + CLS), 768 dim
    tokens = torch.randn(1, 197, 768)
    out = transform(tokens)
    assert out.shape == (1, 768, 14, 14), \
        f"Expected (1, 768, 14, 14), got {out.shape}"


def test_vit_reshape_transform_perfect_square():
    """Verify ViTReshapeTransform handles perfect-square token counts directly."""
    transform = ViTReshapeTransform(patch_size=16, img_size=224)
    tokens = torch.randn(1, 196, 768)  # Already 14×14, no CLS
    out = transform(tokens)
    assert out.shape == (1, 768, 14, 14)

