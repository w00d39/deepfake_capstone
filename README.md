# Single-Frame Deepfake Detection Using Transfer Learning with Vision Transformers and Explainability

**Sarah Wood · Indiana University · Undergraduate Capstone 2026**

## Overview

This project builds a deepfake detection system that not only classifies manipulated facial images but explains its reasoning through attention map visualization. Three deep learning models were trained and compared on the Celeb-DF-v2 benchmark dataset, progressing from a CNN baseline to an explainable Vision Transformer.

The core contribution is pairing transformer-based detection with built-in explainability, enabling human-centered content moderation where a model's reasoning can be visually evaluated rather than blindly trusted.

## Results

| Metric | EfficientNet-B0 | ViT v1 | ViT v2 (Best) |
|---|---|---|---|
| AUC-ROC | 0.77 | 0.81 | **0.83** |
| Macro F1 | 0.70 | 0.72 | **0.75** |
| Real Recall | 0.52 | 0.58 | **0.74** |
| Fake Recall | 0.86 | 0.85 | 0.77 |

Real recall improved from 52% to 74% across the three-model progression, cutting false positives (real images incorrectly flagged as fake) by 47%.

## Attention Map Explainability

The ViT's self-attention mechanism provides a built-in window into the model's reasoning. Attention maps show which facial regions the model examines when making predictions:

- **Fakes:** Attention spreads broadly across facial boundaries where blending artifacts appear
- **Reals:** Attention focuses tightly on specific facial features for verification
- **Errors:** False positives show attention on non-facial elements (background text, logos); false negatives show narrow, unfocused attention

## Repository Structure

```
├── frame_extraction.py      # Frame extraction from Celeb-DF-v2 videos
├── frame_zip.py              # Utility for zipping extracted frames
├── week1.ipynb               # Dataset setup, validation, class distribution
├── week2.ipynb               # Baseline CNN (EfficientNet-B0)
├── week3.ipynb               # Vision Transformer v1 (ViT-Base/16)
├── week4.ipynb               # ViT v2 (tuning + ablation experiment)
├── week5.ipynb               # Explainability (attention map visualization)
├── Week6.ipynb               # Error & bias analysis, calibration, Platt scaling
├── Week7.ipynb               # Robustness testing (compression, noise, resolution)
├── week8.docx                # Final paper
├── README.md
└── LICENSE
```

## Key Findings

**Ablation experiment:** Label smoothing alone shifted the model's class bias without improving discrimination. The ViT v2 improvement required the synergistic combination of stronger augmentation (RandAugment), label smoothing, and increased model capacity (6 unfrozen blocks).

**Source-level bias:** Celeb-real accuracy is 61.4% vs YouTube-real at 93.6%. The model associates celebrity faces with fakes because those individuals appear far more often in the synthetic training data.

**Calibration:** The model is systematically overconfident. Platt scaling failed across splits due to the val/test distribution mismatch (7:1 vs 1.8:1), demonstrating that post-hoc calibration requires distribution-matched data.

**Robustness:** The model is viable under typical social media compression (JPEG Q=70-85) but fragile under heavy noise or extreme compression (Q < 50). Fake recall degrades faster than real recall under all degradation types.

## Dataset

[Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) (Li et al., 2020)
- 6,529 videos: 590 Celeb-real, 300 YouTube-real, 5,639 Celeb-synthesis
- 250,477 extracted frames at 224x224 (every 10th frame, JPEG Q=90)
- Official test split used for benchmark comparability
- Train: 24,856 real / 170,930 fake (6.88:1)
- Val: 4,307 real / 30,329 fake (7.04:1)
- Test: 7,116 real / 12,939 fake (1.82:1)

The frames are not included in this repository due to size. To reproduce, download Celeb-DF-v2 and run `frame_extraction.py`.

## Environment

- **Training:** Google Colab Pro (T4 for EfficientNet, A100 for ViT)
- **Framework:** PyTorch + timm
- **Key libraries:** torchvision, scikit-learn, matplotlib, OpenCV

## Running the Notebooks

1. Download Celeb-DF-v2 and extract frames using `frame_extraction.py`
2. Zip the frames folder and upload to Google Drive
3. Each notebook begins with mounting Drive and unzipping frames to Colab's local disk
4. Notebooks are designed to run sequentially (Week 1 through Week 7)
5. Model checkpoints are saved to Google Drive between sessions

## References

- Groh, M., Epstein, Z., Firestone, C., & Picard, R. (2022). Deepfake detection by human crowds, machines, and machine-informed crowds. *PNAS*, 119(1), e2110013119.
- Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A large-scale challenging dataset for DeepFake forensics. *CVPR*, pp. 3207-3216.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*, pp. 248-255.

## License

MIT
