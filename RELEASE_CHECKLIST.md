# Public-release evidence checklist

Do not state that this release generated the paper's numerical results until
every item below is present and verified.

- [ ] `outputs/splits/busi_seed42.json` records the stratified BUSI split.
- [ ] `outputs/results/classification_metrics_raw.csv` has one row per held-out image, model, and XAI method (20 x 4 x 6 = 480 rows).
- [ ] `outputs/results/model_performance.csv` and the paper figures were generated from the same run.
- [ ] `outputs/results/run_metadata.json` records commit, hardware, package versions, inputs, configuration, and checkpoint hashes.
- [ ] Valid checkpoint links and SHA-256 values are published in `CHECKPOINTS.md`.
- [ ] ISIC-2018/DRIVE stress-test data, code, raw results, and caveats are included.
- [ ] Model/label-randomization controls are included with raw outputs.
- [ ] README, CITATION.cff, and package metadata contain final public author and repository information.

Run `python scripts/release_audit.py --strict` before tagging a public release.
