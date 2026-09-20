# Data and checkpoint policy

This repository intentionally contains no patient images, derived masks, model
weights, API tokens, or experiment outputs. These assets are either governed by
their dataset licences, too large for source control, or both.

## Dataset layout

Place the datasets under `data/` exactly as follows. The dataset directory is
ignored by Git.

```
data/
├── BUSI/
│   ├── benign/       # image files only; BUSI mask files are skipped
│   ├── malignant/
│   └── normal/
└── Kvasir-SEG/
    ├── images/       # JPEG images
    └── masks/        # same filenames as images
```

The paper used a fixed seed (`42`) and 70/15/15 train/validation/test splits.
Before reproducing a number, save the exact dataset release, file manifest, and
the generated split indices alongside the experiment output. Do not claim an
exact numerical match if the source dataset has changed.

## Weights

All checkpoints are written to `outputs/checkpoints/` and are ignored by Git.
The SAM ViT-B checkpoint must be supplied separately and its location set in
`configs/base.yaml`. Download only from the official Segment Anything release
and record its SHA-256 in the experiment log.

The training helper can resume from a locally present checkpoint. It may also
use the configured Google Drive checkpoint IDs as convenience mirrors. A public
release should publish immutable checkpoint links and checksums before relying
on them.
