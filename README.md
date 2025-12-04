# Multi-Organ Abdominal CT Segmentation

Deep learning project for automated segmentation of liver, kidneys, and spleen in abdominal CT scans.

## Project Overview

This project implements and compares five deep learning architectures for multi-organ segmentation:
- **U-Net (2D)** - Baseline slice-based approach
- **Attention U-Net (2D)** - Enhanced with attention mechanisms
- **3D U-Net** - Volumetric processing
- **V-Net** - 3D with residual connections
- **SegResNet** - State-of-the-art 3D model

## Project Structure
```
abdominal_segmentation_project/
├── data/
│   ├── raw/              # Raw CT data (361 image-mask pairs)
│   ├── processed/        # Preprocessed .npy files
│   └── splits/           # Train/val/test split files
├── src/
│   ├── preprocessing/    
│   │   ├── preprocessing.py       # Core preprocessing pipeline
│   │   ├── process_dataset.py     # Batch processing script
│   │   ├── create_splits.py       # Data splitting
│   │   └── preprocess_test_cases.py  # Preprocess specific test cases
│   ├── models/          
│   │   ├── unet_2d.py            # U-Net architecture
│   │   └── losses.py             # Dice + CE loss
│   ├── utils/           
│   │   └── dataset.py            # PyTorch dataset with augmentation
│   ├── visualization/
│   │   └── visualization_suite.py # Enhanced visualizations and comparisons
│   ├── train_unet.py             # Training script
│   ├── validate.py               # Validation script
│   ├── metrics.py                # Comprehensive metrics (Dice, IoU, Hausdorff)
│   ├── error_analysis.py         # Failure case analysis
│   ├── evaluate.py               # Model evaluation utilities
│   └── run_evaluation.py         # Main evaluation runner
├── config/
│   ├── unet_train_config.json    # Training configuration
│   └── test_eval_config.json     # Evaluation configuration
├── results/             # Checkpoints, logs, visualizations
├── notebooks/           
│   └── 01_data_exploration.ipynb # Data analysis notebook
├── docs/                
│   └── DATA_PREPROCESSING.md     # Preprocessing documentation
├── scripts/
│   └── train_baseline.sh         # Training convenience script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Data Setup

### Dataset Information
- **Source**: [FLARE21 Challenge - Subtask1](https://zenodo.org/records/5903037)
- **Format**: 361 training cases
- **Download**: 
  ```bash
  wget "https://zenodo.org/records/5903037/files/Subtask1.zip?download=1" -O Subtask1.zip
  unzip Subtask1.zip
  ```
- **Structure**:
  ```
  Subtask1/
  ├── TrainImage/  (train_0001_0000.nii.gz to train_0361_0000.nii.gz)
  └── TrainMask/   (train_0001.nii.gz to train_0361.nii.gz)
  ```

### Data Splits
The project uses fixed splits for fair comparison across all models:

| Split | File | Cases |
|-------|------|-------|
| Train | `data/splits/train_cases.json` | 252 |
| Val | `data/splits/val_cases.json` | 54 |
| Test | `data/splits/test_cases.json` | 55 |

**Important**: All models must use these same splits for valid comparison.

## Setup Instructions

### 1. Environment Setup
```bash
conda create -n abdomen_seg python=3.9
conda activate abdomen_seg

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### 2. Preprocess Dataset
```bash
cd src/preprocessing

# Process all 361 cases
python process_dataset.py \
    --input_dir /path/to/Subtask1 \
    --output_dir ../../data/processed
```

To preprocess only test cases:
```bash
python preprocess_test_cases.py \
    --input_dir /path/to/Subtask1 \
    --output_dir ../../data/processed \
    --splits_file ../../data/splits/test_cases.json
```

### 3. Train 2D U-Net (Baseline)
```bash
python src/train_unet.py --config config/unet_train_config.json
```

### 4. Train 3D Models

The project includes three optimized 3D architectures for volumetric segmentation:

```bash
# Profile memory usage first (recommended)
python src/profile_models.py --quick --output results

# Train 3D U-Net
python src/train_3d.py --config config/unet3d_config.json

# Train V-Net
python src/train_3d.py --config config/vnet_config.json

# Train SegResNet
python src/train_3d.py --config config/segresnet_config.json

# Or train all 3D models sequentially (PowerShell)
.\scripts\train_all_3d.ps1

# Or train all 3D models sequentially (Bash)
./scripts/train_all_3d.sh
```

### 5. Monitor Training
```bash
tensorboard --logdir results/unet_baseline/logs      # 2D U-Net
tensorboard --logdir results/unet3d_baseline/logs    # 3D U-Net
tensorboard --logdir results/vnet_baseline/logs      # V-Net
tensorboard --logdir results/segresnet_baseline/logs # SegResNet
```

## 3D Model Architectures

### 3D U-Net (`src/models/unet_3d.py`)
- **Type**: Encoder-decoder with skip connections
- **Features**: 
  - Gradient checkpointing for memory efficiency
  - Instance normalization for small batch sizes
  - LeakyReLU activations
  - Optional deep supervision
- **Parameters**: ~5M (base_filters=32)
- **Memory**: ~8GB for batch_size=4, patch_size=(64,128,128)

### V-Net (`src/models/vnet.py`)
- **Type**: Fully convolutional with residual connections
- **Features**:
  - PReLU activations
  - 5×5×5 convolutions in residual blocks
  - Progressive channel expansion (16→32→64→128→256)
  - Deep supervision support
- **Parameters**: ~45M (base_filters=16)
- **Memory**: ~12GB for batch_size=3, patch_size=(64,128,128)

### SegResNet (`src/models/segresnet.py`)
- **Type**: Encoder-decoder with GroupNorm
- **Features**:
  - Group normalization (stable for batch_size≥2)
  - Configurable residual blocks per stage
  - Optional VAE regularization branch
  - Aggressive gradient checkpointing
- **Parameters**: ~10M (init_filters=32)
- **Memory**: ~9GB for batch_size=4, patch_size=(64,128,128)

## 3D Data Pipeline

The 3D dataset (`src/utils/dataset_3d.py`) includes:

- **Patch Extraction**: Configurable patch size (default: 64×128×128)
- **Smart Sampling**: 70% foreground-biased patch selection
- **Memory-Efficient Loading**: LRU cache for volumes
- **3D Augmentations**:
  - Random flips (all axes)
  - 3D rotations (±15°)
  - Elastic deformations
  - Intensity augmentations (brightness, contrast, gamma)
  - Gaussian noise and blur

## Hardware Requirements for 3D Models

### Minimum (for batch_size=2)
- GPU: 16GB VRAM (e.g., NVIDIA V100-16GB, RTX 4080)
- RAM: 64GB
- CPU: 8+ cores

### Recommended (optimized configuration)
- GPU: 32GB VRAM (e.g., NVIDIA L4, A100-40GB)
- RAM: 256GB
- CPU: 64 cores
- Configuration used: g6.16xlarge [L4]

### Memory Profiling
Run the profiling script to find optimal batch sizes:
```bash
python src/profile_models.py --output results
```

This generates:
- `results/profiling_report.json` - Detailed metrics
- `results/PROFILING_REPORT.md` - Human-readable summary

## Evaluation Framework

The project includes a comprehensive evaluation framework for comparing all models.

### Running 2D Evaluation

```bash
# Evaluate U-Net on test set
python src/run_evaluation.py \
    --model unet \
    --config config/unet_train_config.json \
    --checkpoint results/unet_baseline/checkpoints/best_checkpoint.pth \
    --split test \
    --output_dir results/evaluation_test

# Skip Hausdorff distance for faster evaluation
python src/run_evaluation.py \
    --model unet \
    --checkpoint results/unet_baseline/checkpoints/best_checkpoint.pth \
    --split test \
    --no_hausdorff
```

### Running 3D Evaluation

```bash
# Evaluate a single 3D model
python src/evaluate_3d.py \
    --checkpoint results/unet3d_baseline/checkpoints/best_checkpoint.pth \
    --data_dir data/processed \
    --test_split data/splits/test_cases.json \
    --output_dir results/evaluation_3d

# Compare all trained 3D models
python src/evaluate_3d.py --compare \
    --data_dir data/processed \
    --test_split data/splits/test_cases.json \
    --output_dir results/evaluation_3d
```


### Metrics Computed
- **Dice Score** - Per-organ overlap measure
- **IoU (Jaccard Index)** - Intersection over union
- **Pixel Accuracy** - Overall and per-class accuracy
- **Hausdorff Distance (95th percentile)** - Boundary accuracy
- **Statistical Tests** - Paired t-tests, Wilcoxon signed-rank

### Evaluation Outputs
```
results/evaluation_test/
├── evaluation_summary.json    # All metrics
├── error_analysis/
│   ├── error_analysis_report.txt
│   └── confusion_matrix.npy
└── visualizations/
    ├── prediction_grid.png
    └── error_maps/
```

## Preprocessing Pipeline

1. **HU Windowing**: Level=40, Width=400 (abdominal soft tissue)
2. **Normalization**: Intensities to [0, 1]
3. **Resampling**: Isotropic 1x1x1 mm³ spacing
4. **Organ Labels**: 
   - 0: Background
   - 1: Liver
   - 2: Kidneys (merged left+right)
   - 3: Spleen

## Model Architecture

### U-Net (2D) - Baseline
- **Encoder**: 5 levels (64→128→256→512→1024 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 4 classes (background + 3 organs)
- **Parameters**: ~31M

## Training Configuration

```json
{
  "batch_size": 16,
  "epochs": 100,
  "learning_rate": 0.0001,
  "loss": "Dice (50%) + Cross Entropy (50%)",
  "optimizer": "Adam with weight decay (1e-5)",
  "scheduler": "ReduceLROnPlateau (patience=5)",
  "early_stopping": "15 epochs patience"
}
```

## Results

### U-Net 2D Baseline (Validation Set)
| Organ | Dice Score |
|-------|------------|
| Overall | 96.35% |
| Liver | 94.73% |
| Kidneys | 94.33% |
| Spleen | 100.00% |

## References

1. Ronneberger et al., "U-Net: Convolutional networks for biomedical image segmentation," MICCAI 2015
2. Oktay et al., "Attention U-Net: Learning where to look for the pancreas," Medical Image Analysis 2018
3. Çiçek et al., "3D U-Net: Learning dense volumetric segmentation," MICCAI 2016
4. Milletari et al., "V-Net: Fully convolutional neural networks," 3DV 2016
5. Myronenko, "3D MRI brain tumor segmentation using autoencoder regularization," BrainLes 2018