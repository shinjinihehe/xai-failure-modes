# Camera-ready reported results

This file records the family-level BUSI classification results reported in the
accepted camera-ready paper. It is provided as a convenient reference for the
reported experimental results.

| Method | CNN Ins./Del. | Transformer Ins./Del. | VLM Ins./Del. |
|---|---:|---:|---:|
| Grad-CAM | 0.69 / 0.45 | 0.75 / 0.24* | 0.51 / 0.35* |
| Grad-CAM++ | 0.68 / 0.45 | 0.74 / 0.25* | 0.50 / 0.36* |
| Integrated Gradients | 0.65 / 0.35 | 0.72 / 0.30 | 0.52 / 0.37 |
| LIME | 0.63 / 0.89 | 0.69 / 0.94 | 0.48 / 0.45 |
| SHAP | 0.64 / 0.37 | 0.70 / 0.32 | 0.50 / 0.39 |
| Occlusion | 0.66 / 0.43 | 0.81 / 0.28 | 0.55 / 0.34 |

Lower deletion AUC is better. `*` denotes the patch-token reshape
approximation used for non-convolutional backbones. The paper also reports CNN
LIME deletion AUC values of 0.84 for ResNet-50 and 0.94 for DenseNet-121.
