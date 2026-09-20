"""Validate a clone before starting an expensive experiment.

This performs no downloads and does not require datasets. It catches common
configuration/path errors that otherwise surface hours into a run.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import get_config  # noqa: E402


def main() -> int:
    config = get_config()
    errors: list[str] = []
    if config.training.classification.epochs != 50:
        errors.append("classification epochs must be 50 for the camera-ready protocol")
    if config.training.segmentation.epochs != 100:
        errors.append("segmentation epochs must be 100 for the camera-ready protocol")
    if config.models.segmentation.sam_adapter.epochs != 15:
        errors.append("SAM adapter epochs must be 15 (reproducibility.md §Protocol)")
    if config.xai.evaluation_subset.classification != 20:
        errors.append("classification XAI subset must contain 20 held-out images")
    if config.data.busi.img_size != 224:
        errors.append("BUSI img_size must be 224 (required for ViT-B/16 patch reshape transform)")
    if config.project.seed != 42:
        errors.append("project seed must be 42 to match the published BUSI split manifest")
    for required in ("data_root", "outputs_root", "checkpoints_dir", "results_dir"):
        value = Path(getattr(config.paths, required))
        if not value.is_absolute():
            errors.append(f"{required} is not absolute: {value}")
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Configuration is internally consistent.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Datasets and checkpoints are intentionally not checked here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
