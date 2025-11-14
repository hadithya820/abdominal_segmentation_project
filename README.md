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
│   │   └── create_splits.py       # Data splitting
│   ├── model/          
│   │   ├── unet_2d.py            # U-Net architecture
│   │   └── losses.py             # Dice + CE loss
│   ├── utils/           
│   │   └── dataset.py            # PyTorch dataset with augmentation
│   ├── train_unet.py    # Training script
│   └── validate.py      # Validation script
├── config/
│   └── unet_train_config.json    # Training configuration
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
- **Source**: [FLARE21 Challenge - Subtask1](https://zenodo.org/records/5903037/files/Subtask1.zip?download=1)
- **Format**: Subtask1 (361 training cases)
- **Download**: 
  ```bash
  # Download dataset
  wget https://zenodo.org/records/5903037/files/Subtask1.zip?download=1 -O Subtask1.zip
  
  # Extract
  unzip Subtask1.zip
  ```
- **Data Location**: `/dbfs/tmp/html_output/data` (Update this placeholder!)
- **Structure**:
  ```
  Subtask1/
  ├── TrainImage/  (train_0001_0000.nii.gz to train_0361_0000.nii.gz)
  └── TrainMask/   (train_0001.nii.gz to train_0361.nii.gz)
  ```

### Update Data Path
**IMPORTANT**: Update the data path in your configuration:

1. Open `config/unet_train_config.json`
2. Update `data_dir` to point to your processed data location
3. For preprocessing, update `process_dataset.py` input path

## Setup Instructions

### 1. Environment Setup
```bash
# Create conda environment
conda create -n abdomen_seg python=3.9
conda activate abdomen_seg

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision

# Install other requirements
pip install -r requirements.txt
```

### 2. Preprocess Dataset
```bash
cd src/preprocessing

# Process all 361 cases (adjust input_dir to your data location)
python process_dataset.py \
    --input_dir /dbfs/tmp/html_output/data \
    --output_dir ../../data/processed
```

**Output**: 
- CT volumes: `{case_id}_ct.npy` (float32, normalized to [0,1])
- Segmentation masks: `{case_id}_seg.npy` (uint8, labels: 0-3)
- Metadata: `{case_id}_metadata.json`

### 3. Create Data Splits
```bash
python create_splits.py
```

**Default split**: 70% train / 15% val / 15% test (253/54/54 cases)

### 4. Train Baseline U-Net
```bash
cd ../..

# Using config file
python src/train_unet.py --config config/unet_train_config.json

# Or use convenience script
bash scripts/train_baseline.sh
```

### 5. Monitor Training
```bash
# In a separate terminal
tensorboard --logdir results/unet_baseline/logs
```

Open browser to `http://localhost:6006`

### 6. Validate Model
```bash
python src/validate.py \
    --config config/unet_train_config.json \
    --checkpoint results/unet_baseline/checkpoints/best_checkpoint.pth
```

### 7. Visualize Results
```bash
# Generate grid of prediction samples
python src/visualize_results.py \
    --config config/unet_train_config.json \
    --checkpoint results/unet_baseline/checkpoints/best_checkpoint.pth \
    --mode grid \
    --num_samples 8 \
    --output_dir results/visualizations

# Visualize single sample
python src/visualize_results.py \
    --config config/unet_train_config.json \
    --checkpoint results/unet_baseline/checkpoints/best_checkpoint.pth \
    --mode single \
    --sample_idx 100

# Visualize full case (all slices)
python src/visualize_results.py \
    --config config/unet_train_config.json \
    --checkpoint results/unet_baseline/checkpoints/best_checkpoint.pth \
    --mode case \
    --case_id case_00001
```

### 8. Plot Training History
```bash
# Generate all training plots
python src/plot_training_history.py \
    --log_dir results/unet_baseline/logs \
    --output_dir results/training_plots \
    --plot_type all

# Generate specific plots
python src/plot_training_history.py \
    --log_dir results/unet_baseline/logs \
    --output_dir results/training_plots \
    --plot_type losses  # Options: losses, lr, summary, all
```

## Preprocessing Pipeline

### Steps:
1. **HU Windowing**: Level=40, Width=400 (abdominal soft tissue window)
2. **Normalization**: Intensities → [0, 1]
3. **Resampling**: → Isotropic 1×1×1 mm³ spacing
4. **Organ Filtering**: 
   - Label 0: Background
   - Label 1: Liver
   - Label 2: Kidneys (merged left+right)
   - Label 3: Spleen

**Details**: See `docs/DATA_PREPROCESSING.md`

## Data Augmentation

### Training:
- Random rotation: ±15° (p=0.5)
- Horizontal/vertical flips (p=0.5/0.3)
- Elastic deformations (p=0.3)
- Brightness/contrast adjustments (p=0.5)
- Gaussian noise (p=0.3)

### Validation/Test:
- Normalization only (no augmentation)

## Model Architecture

### U-Net (2D) - Baseline
- **Encoder**: 5 levels (64→128→256→512→1024 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 1×1 conv to 4 classes (background + 3 organs)
- **Parameters**: ~31M
- **Input**: 2D axial CT slices
- **Output**: Per-pixel class probabilities

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

## File Configurations

### Update These Paths:
1. **Raw data**: Update in preprocessing commands
   ```bash
   --input_dir /dbfs/tmp/html_output/data  # Your actual data path
   ```

## References

1. Ronneberger et al., "U-Net: Convolutional networks for biomedical image segmentation," MICCAI 2015
2. Oktay et al., "Attention U-Net: Learning where to look for the pancreas," Medical Image Analysis 2018
3. Çiçek et al., "3D U-Net: Learning dense volumetric segmentation," MICCAI 2016
4. Milletari et al., "V-Net: Fully convolutional neural networks," 3DV 2016
5. Myronenko, "3D MRI brain tumor segmentation using autoencoder regularization," BrainLes 2018