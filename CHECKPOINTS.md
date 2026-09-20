# Checkpoint manifest

Publish one immutable URL and SHA-256 digest per final model checkpoint before
asserting that a public clone reproduces paper results.

| Model | File | Immutable URL | SHA-256 | BUSI/Kvasir test metric |
|---|---|---|---|---|
| ResNet-50 | `resnet50_best.pth` | Pending release | Pending | Accuracy / macro-F1 |
| DenseNet-121 | `densenet121_best.pth` | Pending release | Pending | Accuracy / macro-F1 |
| ViT-B/16 | `vit_b16_best.pth` | Pending release | Pending | Accuracy / macro-F1 |
| BiomedCLIP | `biomedclip_best.pth` | Pending release | Pending | Accuracy / macro-F1 |
| U-Net | `unet_best.pth` | Pending release | Pending | Dice / IoU |
| Transformer segmenter | `transunet_best.pth` | Pending release | Pending | Dice / IoU |
| SAM Adapter | `sam_adapter_best.pth` | Pending release | Pending | Dice / IoU |

Do not use browser error pages or mutable sharing links as checkpoints. Verify
each download with `sha256sum` before publishing its row.
