# Skin Cancer Classification — Transfer Learning Benchmark

## Overview

This project evaluates deep learning and machine learning approaches for classifying skin lesions into 9 diagnostic categories using the ISIC Skin Cancer dataset. It addresses three comparative analyses:

1. Performance of pretrained CNN architectures fine-tuned via transfer learning
2. Performance of classical machine learning classifiers trained on deep features extracted from a CNN backbone
3. Computational efficiency of each CNN architecture

The implementation is provided as a Google Colab notebook (`skin_cancer_transfer_learning.ipynb`) so it can be run without any local setup.

**Dataset:** Skin Cancer ISIC — The International Skin Imaging Collaboration (9 classes)
Source: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

## Results Produced

### Table 1 — Comparison of Transfer Learning Models

Eight pretrained architectures are fine-tuned on the dataset and evaluated on a held-out test set:

AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0

Each model is scored on Accuracy, Precision, Recall, F1-Score, and AUC (all reported as macro-averaged percentages, since this is a multi-class problem).

### Table 2 — Comparison of Different Classifiers

Deep features are extracted from a trained CNN backbone (ResNet50, by default) and used to train seven classical classifiers:

Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Linear SVM, RBF-SVM, XGBoost

The same five metrics are reported for each classifier.

### Table 3 — Computational Efficiency Comparison

For each CNN architecture, the following are measured: parameter count (millions), saved model size (MB), computational cost (GFLOPs), and average inference time per image (ms), alongside its classification accuracy for reference.

*Note: the original task specification lists seven models for Table 3 (excluding ResNet101). The notebook profiles all eight models used in Table 1 for internal consistency; the ResNet101 row can be removed from the output if an exact match to the specification is required.*

## Output Files

| File | Description |
|---|---|
| `table1_models.csv` | Transfer learning model comparison |
| `table2_classifiers.csv` | Classical classifier comparison |
| `table3_efficiency.csv` | Computational efficiency comparison |
| `skin_cancer_results.xlsx` | All three tables combined, one sheet each — downloads automatically when the notebook finishes |

## Requirements

- A Google account (to run Colab)
- A Kaggle account (to access the dataset)
- Colab runtime set to GPU (Runtime → Change runtime type → GPU)

## Setup Instructions

### Running the notebook

1. Upload `skin_cancer_transfer_learning.ipynb` to Google Colab.
2. Set the runtime type to GPU.
3. Run all cells in order (Runtime → Run all).
4. On completion, `skin_cancer_results.xlsx` is downloaded automatically, containing all three result tables.

## Notebook Structure

| Section | Purpose |
|---|---|
| 1. Install dependencies | Installs required packages (`kaggle`, `torchmetrics`, `xgboost`, `thop`, `scikit-learn`) |
| 2. Download dataset | Retrieves and extracts the dataset from Kaggle |
| 3. Configuration | Defines file paths and training hyperparameters |
| 4. Data loaders | Builds train/validation/test splits with appropriate image transforms |
| 5. Model definitions | Loads each pretrained architecture with a 9-class classification head |
| 6. Training and evaluation functions | Fine-tuning loop and metric calculation |
| 7. Table 1 generation | Trains and evaluates all eight CNN models |
| 8. Table 2 generation | Extracts deep features and trains classical classifiers |
| 9. Table 3 generation | Profiles parameters, model size, FLOPs, and inference time |
| 10. Export | Writes results to Excel and triggers download |

## Configurable Parameters

| Parameter | Location | Default | Description |
|---|---|---|---|
| `EPOCHS` | Section 3 | 10 | Number of training epochs per model |
| `FEATURE_BACKBONE` | Section 8 | `"ResNet50"` | CNN used to generate deep features for Table 2 |
| `IMG_SIZE` | Section 3 | 224 | Input image resolution |
| `BATCH_SIZE` | Section 3 | 32 | Training batch size |
| `LR` | Section 3 | 1e-4 | Learning rate |

For an initial test run, reducing `EPOCHS` (e.g., to 2–3) is recommended to verify the pipeline executes correctly before committing to the full training run, which can take a significant amount of time across all eight models and seven classifiers, particularly on Colab's free-tier GPU.

## Troubleshooting

If the dataset extracts into a folder name different from what is expected, the output of the `print(glob.glob(...))` statement in Section 2 will show the actual directory structure. Update `DATA_ROOT` in Section 3 accordingly.
