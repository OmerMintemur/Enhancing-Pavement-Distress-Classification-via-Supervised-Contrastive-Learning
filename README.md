# Enhancing Pavement Distress Classification via Supervised Contrastive Learning

A hybrid training strategy that combines **Cross-Entropy (CE)** and **Supervised Contrastive Learning (SupCon)** losses to improve discriminative feature learning for pavement distress classification under data-scarce conditions with high inter-class visual similarity.

## Overview

Deep learning models for pavement distress detection are often limited by scarce data and visually similar classes. This work integrates a **projection head** on top of standard CNN backbones (**ResNet50** and **EfficientNetB0**) and trains them with a joint objective:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{SupCon}$$

The Cross-Entropy branch optimizes class separation, while the Supervised Contrastive branch pulls same-class samples together and pushes different classes apart in the embedding space, yielding more discriminative representations.

## Method

The model performs two learning objectives simultaneously:

- **Classification branch** — the feature vector from the backbone is passed through the encoder MLP and a linear classifier, optimized with Cross-Entropy loss.
- **Supervised Contrastive branch** — the same feature vector is passed through a projection head and L2-normalized to produce embeddings (`z`), used in the SupCon loss. Unlike SimCLR, which treats only augmented views of the same image as positives, SupCon treats **all samples of the same class** as positive pairs.

## Results

Evaluated on the **PaveDistress** dataset (3 classes: *Crack*, *Patch*, *Other*), the proposed method outperforms conventional fine-tuning:

| Model          | Fine-Tuning (Acc) | + SupCon (Acc) | Gain    |
| -------------- | ----------------- | -------------- | ------- |
| ResNet50       | 90.77%            | 91.85%         | +1.08%  |
| EfficientNetB0 | 91.11%            | 92.74%         | +1.63%  |

t-SNE analysis confirms improved class separability, with Silhouette scores rising from 0.476 → 0.550 (ResNet50) and 0.471 → 0.521 (EfficientNetB0).

### Hyperparameters

| Parameter            | Value              |
| -------------------- | ------------------ |
| Input size           | 224 × 224          |
| Epochs               | 100                |
| Batch size           | 64                 |
| Optimizer            | AdamW              |
| Learning rate        | 3 × 10⁻⁴           |
| Temperature (τ)      | 0.07               |
| λ (SupCon weight)    | 0.1                |

## Repository Contents

| File       | Description                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------- |
| `Main.py`  | Training script implementing the joint CE + SupCon objective, two-view augmentation, AMP, and OneCycleLR scheduling. |
| `Test.py`  | Evaluation script reporting Accuracy, F1, Precision, and Recall on the train/validation sets.     |

## Dataset Structure

Place the dataset one level above the scripts, organized in `ImageFolder` format:

```
dataset/
├── train/
│   ├── Crack/
│   ├── Patch/
│   └── Other/
└── val/
    ├── Crack/
    ├── Patch/
    └── Other/
```

## Requirements

```
torch
torchvision
scikit-learn
```

Install with:

```bash
pip install torch torchvision scikit-learn
```

## Usage

**Training** — runs 3 repeated experiments and saves the best model per run based on validation accuracy:

```bash
python Main.py
```

Checkpoints and training logs are written to `my_experiment_logs<backbone>_Run_<n>/`.

**Evaluation** — load a trained checkpoint and report metrics:

```bash
python Test.py
```

> Update `model_folder` in `Test.py` to point to the run you want to evaluate.

## Citation


