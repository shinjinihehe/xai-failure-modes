# Reproducibility protocol

The camera-ready protocol trains four BUSI classifiers for 50 epochs and three
Kvasir-SEG segmenters for 100 epochs (SAM adapter: 15 epochs), with AdamW,
weight decay `1e-4`, and cosine scheduling. Classification uses batch size 16
and peak learning rate `3e-5`; segmentation uses batch size 8 and peak learning
rate `1e-4`. The configuration encodes those settings.

Classification faithfulness is a diagnostic evaluation over 20 held-out images
per model. For each image and XAI method, compute insertion AUC, deletion AUC,
and sparsity, then aggregate only after preserving the per-image records.
Do not overwrite per-image results with the paper table: the table is a summary,
not raw experimental data.

The segmentation arm is a structural usability audit. It produces qualitative
maps and reports Dice/IoU task performance; it is not evidence for a
standardised dense-pixel faithfulness benchmark. This distinction follows the
camera-ready limitations and should remain explicit in derivative work.

## Required run record

For each run, save `outputs/results/run_metadata.json` with the Git commit,
Python/package versions, hardware, input-file manifest hash, configuration,
random seed, checkpoint SHA-256 values, and command line. Store raw metrics as
one row per image/model/method before producing family-level figures.

## Interpretation guardrails

Grad-CAM and Grad-CAM++ on ViT-B/16 and BiomedCLIP use a patch-token spatial
reshape. These maps are marked as an approximation and must not be interpreted
as conventional convolutional Grad-CAM. LIME uses quickshift superpixels, so
pixel-level deletion has tied ranks; report its hyperparameters with every run.

## Architecture implementation note

The paper refers to the Transformer segmentation model as "TransUNet" in Table I
and the text. In this repository the `transunet` experiment key is implemented
using a **SegFormer/MiT-B2** encoder-decoder via `segmentation-models-pytorch`
(`smp.Segformer`), which is a Transformer-based segmentation architecture with
convolutional decoder layers, matching the paper's key claim that "Grad-CAM
applies to TransUNet because its decoder retains convolutional layers." The Dice
and IoU metrics in Table I (0.874 / 0.804) correspond to this implementation.
Reproduction attempts should use SegFormer/MiT-B2 via the `smp.Segformer` API.
