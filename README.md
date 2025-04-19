# DIET: Dynamic Information Exchange Training

## Overview

This contains the implementation of DIET (Dynamic Information Exchange Training), a novel fine-tuning approach that enhances the zero-shot capabilities of pre-trained vision models. DIET uses a specialized training method that combines traditional supervised learning with a secondary objective that encourages models to maintain and improve their representation capabilities.

## Key Concepts

DIET introduces several key concepts:

1. **Dual Head Architecture**: The model uses two classification heads:
   - A standard probe head for the primary classification task
   - A DIET head trained on randomly assigned "diet classes"

2. **Dynamic Loss Weighting**: The relative importance of the DIET loss and the probe loss changes during training:
   - Early training: Higher weight on DIET loss (0.6 DIET, 0.4 probe)
   - Mid training: Balanced weights (0.4 DIET, 0.6 probe)
   - Late training: Higher weight on probe loss (0.2 DIET, 0.8 probe)

3. **Label Smoothing**: The DIET head uses label smoothing to prevent overfitting to the random diet classes.

4. **Zero-Shot Evaluation**: The model is evaluated using several zero-shot metrics, including k-NN, k-means clustering, and linear probing.

## Repository Structure

```
├── main.py                 # Main entry point for running experiments
├── checkpoints/            # Saved model checkpoints
├── config/                 # Configuration files
│   └── params.py           # Hyperparameter configurations
├── datasets/               # Dataset handling
│   └── data_loader.py      # Data loading utilities
├── evaluation/             # Evaluation utilities
│   ├── metrics.py          # Zero-shot metrics implementation
│   └── experiment_results.py # Results processing
├── models/                 # Model implementations
│   ├── dinov2.py           # DINOv2 model wrapper
│   ├── mae.py              # MAE model wrapper
│   ├── ijepa.py            # I-JEPA model wrapper
│   ├── mambavision.py      # MambaVision model wrapper
│   └── aim.py              # AIM model wrapper
├── results/                # Experiment results
├── training/               # Training utilities
│   ├── trainer.py          # DIET Trainer implementation
│   └── config.py           # Training configurations
└── utils/                  # Utility functions
    └── wandb_logger.py     # Weights & Biases logging utilities
      __ sanity_check.py      # Sanity Check 
```

## Supported Models

The framework supports multiple state-of-the-art pre-trained vision models:

1. **DINOv2** (Facebook AI): Self-supervised vision transformer with strong zero-shot capabilities
   - Variants: small, base, large, giant

2. **MAE** (Masked Autoencoder): Self-supervised vision transformer that learns by reconstructing masked patches
   - Variants: base, large, huge

3. **I-JEPA** (Joint-Embedding Predictive Architecture): Self-supervised model that predicts features of masked image regions
   - Variants: b16_1k, l14_22k, h14_1k, etc.

4. **MambaVision**: State-space model for vision tasks with selective state space layers
   - Variants: T, S, B, L, L2, etc.

5. **AIM** (Autoregressive Image Models): Autoregressive model for image generation
   - Variants: 600M, 1B, 3B, 7B

6. **ResNet50** (Baseline): Standard ResNet50 pre-trained on ImageNet

## Supported Datasets

The framework supports several image classification datasets:

1. **CIFAR-10**: 10-class image classification dataset
2. **MedMNIST**: Collection of medical image datasets
   - PathMNIST, ChestMNIST, DermaMNIST, OctMNIST, PneumoniaMNIST
3. **PlantNet300K**: Plant species classification dataset
4. **Galaxy10 DECals**: Galaxy morphology classification
5. **Crop14 Balance**: Agricultural crop classification dataset

## Key Parameters

### Model Parameters
- `backbone_type`: Type of backbone architecture ("resnet50", "dinov2", "mae", "ijepa", "mambavision", "aim")
- `model_size`: Size variant of the model ("small", "base", "large", "giant" for DINOv2, etc.)
- `embedding_dim`: Feature dimensionality of the model (automatically set based on model type)
- `projection_dim`: Dimensionality of the projection head (default: 256)

### Dataset Parameters
- `dataset_name`: Name of the dataset to use
- `num_classes`: Number of classes in the dataset (set automatically)
- `num_diet_classes`: Number of random diet classes (default: 100-200, adjust based on dataset)
- `da_strength`: Data augmentation strength (0-3)
- `limit_data`: Number of training samples to use (set to np.inf for full dataset)

### Training Parameters
- `num_epoch`: Number of training epochs (default: 30)
- `batch_size`: Batch size for training (default: 64-128, adjust based on model and hardware)
- `lr`: Learning rate (typically 1e-4 to 5e-4)
- `weight_decay`: Weight decay for regularization (default: 0.05)
- `label_smoothing`: Label smoothing factor for DIET loss (default: 0.3)
- `device`: Device to use for training ("cuda" or "cpu")

## Training Methodology

The DIET training process follows these steps:

1. **Initialization**:
   - Load pre-trained backbone model
   - Create projection head and classification heads
   - Prepare data loaders with diet class assignments

2. **Initial Evaluation**:
   - Perform zero-shot evaluation on the test set before training
   - Measure k-NN accuracy, k-means clustering metrics, and linear probe accuracy

3. **Training Loop**:
   - Forward pass: Extract features from the backbone model
   - Normalize features and pass through projection head
   - Calculate DIET loss and probe loss
   - Apply dynamic loss weighting based on current epoch
   - Update model parameters

4. **Periodic Evaluation**:
   - Evaluate test accuracy 
   - Perform zero-shot evaluation every few epochs
   - Track metrics and create visualizations

5. **Final Evaluation**:
   - Compare final zero-shot metrics to initial metrics
   - Calculate improvement in representation quality

## Zero-Shot Evaluation Metrics

The framework evaluates zero-shot performance using several metrics:

1. **k-NN Accuracy**: Accuracy of k-nearest neighbors classifier using the model's features
2. **k-means ARI**: Adjusted Rand Index of k-means clustering compared to ground truth
3. **k-means NMI**: Normalized Mutual Information of k-means clustering
4. **Linear Probe Accuracy**: Accuracy of a linear classifier trained on frozen features

## Experiment Tracking

The framework uses Weights & Biases (wandb) for experiment tracking:

1. **Training Metrics**:
   - DIET loss and probe loss
   - Training and test accuracy

2. **Zero-Shot Metrics**:
   - k-NN accuracy
   - k-means clustering metrics (ARI, NMI)
   - Linear probe accuracy

3. **Visualizations**:
   - Training progress plots
   - Zero-shot metrics progression
   - Tables comparing initial and final performance

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- torchvision
- transformers
- scikit-learn
- wandb
- PIL
- matplotlib
- numpy
- tqdm

### Running an Experiment

change the parameters within config/params.py to select the parameters that fit your style 

### Using the Notebook

Alternatively, you can use the provided Jupyter notebook (`Dinov2_Mnist_new_version copy 2.ipynb`), which offers an interactive environment for running experiments with detailed visualizations.


## Sanity Checking

The framework includes a sanity checking utility that verifies the zero-shot performance of pre-trained models on CIFAR-10. This helps ensure that the pre-trained models are properly loaded and functioning as expected before applying DIET finetuning.

