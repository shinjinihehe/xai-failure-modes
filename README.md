# IPTA 2026 XAI Benchmarking

**When Explainability Goes Silent: Diagnosing XAI Failure Modes in Medical Imaging**

This repository contains the official codebase for the accepted IPTA 2026 paper on XAI faithfulness evaluation across CNN, Transformer, and VLM architectures in medical imaging. It is being released publicly upon acceptance.

## Overview

- **Task**: Classification (BUSI breast ultrasound) + Segmentation (Kvasir-SEG colonoscopy)
- **Models**: 7 models across 3 architecture families
  - CNN: ResNet-50, DenseNet-121, U-Net
  - Transformer: ViT-B/16, SegFormer (MiT-B2)
  - VLM: BiomedCLIP, SAM Adapter
- **XAI Methods**: 6 methods (Grad-CAM, Grad-CAM++, Integrated Gradients, LIME, SHAP, Occlusion)
- **Metrics**: Classification Insertion/Deletion AUC and Sparsity; segmentation Dice/IoU plus qualitative structural audit

## Quick Start

### Installation

```bash
# Clone and enter
cd ipta2026-xai

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify config and environment (no dataset required)
python scripts/validate_setup.py

# Run protocol compliance unit tests (no dataset required)
pytest tests/
```

### Prepare Data

Organize licensed datasets as follows (they are deliberately Git-ignored):
```
data/
├── BUSI/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── Kvasir-SEG/
    ├── images/
    └── masks/
```

### Run Training

```bash
# Train all 7 models
python scripts/train_all.py
```

### Compute XAI & Evaluate

```bash
# Generate XAI maps, quantitative metrics, and paper figures
python scripts/compute_xai.py
```

### Reproduce the paper's classification protocol

```bash
# 20 held-out images x 4 models x 6 methods = 480 raw metric rows
python scripts/run_classification_benchmark.py
python scripts/release_audit.py --strict
```

The benchmark writes split indices, raw per-image metrics, family summaries,
and run metadata under `outputs/`. It stops if a required checkpoint is absent;
it never substitutes a result from a different model or run.

Cross-dataset and randomization controls are separate, explicit runs:

```bash
python scripts/run_cross_dataset_stress.py --dataset isic2018 --root data/ISIC-2018
python scripts/run_cross_dataset_stress.py --dataset drive --root data/DRIVE
python scripts/run_sanity_checks.py
python scripts/run_label_randomization.py
python scripts/generate_paper_figures.py
```

Generated results are saved to:
- `outputs/results/xai_metrics.csv` - Quantitative metrics
- `outputs/figures/` - All paper figures

## Repository Structure

```
ipta2026-xai/
├── configs/
│   └── base.yaml                        # Complete configuration (hyperparameters, paths, seeds)
├── scripts/
│   ├── train_all.py                     # Train all 7 models (classification + segmentation)
│   ├── compute_xai.py                   # XAI computation & evaluation (single demo image)
│   ├── run_classification_benchmark.py  # 20-image faithfulness protocol → raw CSV
│   ├── run_cross_dataset_stress.py      # BUSI-trained CNN stress test on ISIC-2018 / DRIVE
│   ├── run_sanity_checks.py             # Adebayo weight-randomization control
│   ├── run_label_randomization.py       # Shuffled-label training control
│   ├── generate_paper_figures.py        # Recreate paper figures from raw CSV
│   ├── release_audit.py                 # Fail-closed release evidence check
│   └── validate_setup.py               # Config/environment pre-flight check
├── src/
│   ├── data/
│   │   └── datasets.py                  # BUSI & Kvasir-SEG datasets, stratified splits
│   ├── models/
│   │   ├── classification/
│   │   │   └── models.py                # ResNet-50, DenseNet-121, ViT-B/16, BiomedCLIP
│   │   └── segmentation/
│   │       └── models.py                # U-Net, SegFormer (MiT-B2), SAM Adapter
│   ├── training/
│   │   └── trainer.py                   # Training loops, checkpoint manager
│   ├── xai/
│   │   └── methods.py                   # All 6 XAI methods (with ViT reshape transform)
│   ├── metrics/
│   │   └── evaluation.py                # Insertion/Deletion AUC, Sparsity, Pointing Game
│   └── utils/
│       ├── config.py                    # YAML config loader with dot-notation access
│       ├── provenance.py                # SHA-256 hashing, run metadata, git revision
│       └── visualization.py             # Plotting: XAI grids, heatmaps, bar charts
├── tests/
│   └── test_protocol.py                 # Lightweight unit tests (no dataset needed)
├── docs/
│   ├── reported_results.md              # Camera-ready numbers from Table II
│   ├── reproducibility.md               # Protocol, guardrails, required run record
│   └── data_and_checkpoints.md          # Dataset layout, checkpoint policy

├── outputs/
│   ├── checkpoints/                     # Model weights (git-ignored)
│   ├── results/                         # CSV metrics and run metadata (git-ignored)
│   ├── figures/                         # Paper figures (git-ignored)
│   └── xai_maps/                        # Saved saliency maps (git-ignored)
├── CHECKPOINTS.md                       # Checkpoint manifest (URLs + SHA-256)
├── RELEASE_CHECKLIST.md                 # Pre-release evidence checklist
├── CITATION.cff                         # Machine-readable citation
├── environment.yml                      # Conda environment spec
├── requirements.txt                     # pip requirements
└── setup.py                             # Installable package
```

## ⚙️ Configuration

All hyperparameters, paths, and settings are in `configs/base.yaml`. Key sections:

- **data**: Dataset paths, splits, augmentation, normalization
- **models**: Model architectures, pretrained weights, learning rates
- **training**: Epochs, batch sizes, optimizers, schedulers
- **xai**: Method-specific parameters (LIME samples, SHAP background, etc.)
- **metrics**: Insertion/deletion steps, thresholds
- **visualization**: Figure sizes, DPI

## 🔬 Key Features

1. **Config-driven**: No hardcoded values - everything in YAML
2. **Auto-resume**: Checkpoints automatically loaded, training skipped if exists
3. **Checkpoint resume**: Existing local checkpoints are loaded safely before training
4. **Memory efficient**: Models offloaded to CPU between training/XAI
5. **Reproducible**: Fixed seeds, consistent splits, versioned dependencies

## 📊 Reproducing Paper Results

The paper reports classification evaluation on **20 held-out test images per model**. The complete protocol and interpretation guardrails are in [docs/reproducibility.md](docs/reproducibility.md), and the accepted-paper results are documented in [docs/reported_results.md](docs/reported_results.md).

```bash
# After training, run qualitative XAI generation
python scripts/compute_xai.py
```

This computes:
- Grad-CAM / Grad-CAM++ (with proper ViT reshape transform)
- Integrated Gradients (50 steps)
- LIME (200 samples, quickshift segmentation)
- SHAP KernelExplainer (32 samples, k-means background with k=10)
- Occlusion Sensitivity (16×16 patches, stride 8)

## 📝 Citation

```bibtex
@inproceedings{...,
  title={When Explainability Goes Silent: Diagnosing XAI Failure Modes in Medical Imaging},
  author={Anonymous},
  booktitle={IPTA 2026},
  year={2026}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## Data, weights, and scope

No patient images, model weights, or raw experiment outputs are included. See [docs/data_and_checkpoints.md](docs/data_and_checkpoints.md) for the required dataset layout, checkpoint policy, and an explicit release checklist. The transformer segmentation experiment uses a SegFormer/MiT-B2 encoder-decoder under the `transunet` experiment key.
