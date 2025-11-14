# Data Preprocessing Documentation

## Overview
This document describes the preprocessing pipeline for abdominal CT scans used in the multi-organ segmentation project.

## Dataset Structure

### Supported Formats

#### 1. Subtask1 Format (Default for this project)
```
data/
├── Subtask1/
│   ├── TrainImage/
│   │   ├── train_0001_0000.nii.gz
│   │   ├── train_0002_0000.nii.gz
│   │   └── ... (361 total)
│   └── TrainMask/
│       ├── train_0001.nii.gz
│       ├── train_0002.nii.gz
│       └── ... (361 total)
```

**Data Path Placeholder**: `/dbfs/tmp/html_output/data`
- Update this path in your configuration files to match your data location
- 361 training image-mask pairs available

#### 2. AbdomenCT-1K Format (Alternative)
```
data/
├── Case_00000/
│   ├── imaging.nii.gz
│   └── segmentation.nii.gz
├── Case_00001/
│   └── ...
```

## Preprocessing Pipeline

### Step 1: Hounsfield Unit (HU) Windowing
**Purpose**: Focus on soft tissue contrast in abdominal region

- **Window Level**: 40 HU
- **Window Width**: 400 HU
- **HU Range**: [-160, 240] HU

**Rationale**: 
- Level 40 HU is optimal for liver and soft organs
- Width 400 HU captures liver (40-60 HU), kidneys (30 HU), spleen (50 HU), and surrounding tissues

**Implementation**:
```python
hu_min = level - (width / 2) = 40 - 200 = -160
hu_max = level + (width / 2) = 40 + 200 = 240
ct_windowed = np.clip(ct_data, -160, 240)
```

### Step 2: Intensity Normalization
**Purpose**: Standardize intensity values for neural network training

- **Output Range**: [0, 1]
- **Formula**: `normalized = (windowed - hu_min) / (hu_max - hu_min)`

**Benefits**:
- Stable gradient flow during training
- Consistent across different scanners
- Faster convergence

### Step 3: Resampling to Isotropic Spacing
**Purpose**: Standardize voxel dimensions across all scans

- **Target Spacing**: 1×1×1 mm³ (isotropic)
- **Interpolation**: 
  - CT images: Cubic (order=3) for smooth interpolation
  - Segmentation masks: Nearest neighbor (order=0) to preserve labels

**Original Spacing (typical)**:
- In-plane (x, y): 0.7-0.8 mm
- Slice thickness (z): 3-3.2 mm

**Why Isotropic?**
- Enables fair comparison in all directions
- Better for 3D models
- Consistent feature learning

### Step 4: Organ Label Filtering
**Purpose**: Focus on target organs and simplify label space

**Original Labels** (AbdomenCT-1K):
- Label 0: Background
- Label 1: Liver
- Label 2: Right kidney
- Label 3: Left kidney
- Label 6: Spleen
- (Other organs excluded)

**New Labels**:
- Label 0: Background
- Label 1: Liver
- Label 2: Kidneys (merged left + right)
- Label 3: Spleen

**Rationale**: 
- Merge kidneys to reduce class imbalance
- Focus on 3 major organs as per project scope

## Usage

### Process Dataset
```bash
cd src/preprocessing
python process_dataset.py \
    --input_dir /dbfs/tmp/html_output/data \
    --output_dir ../../data/processed \
    --num_cases 361
```

**Parameters**:
- `--input_dir`: Path to raw data (automatically detects Subtask1 or AbdomenCT-1K format)
- `--output_dir`: Where to save processed .npy files
- `--num_cases`: Optional limit on number of cases (default: all)

**Output Files** (per case):
- `{case_id}_ct.npy`: Preprocessed CT volume (float32)
- `{case_id}_seg.npy`: Preprocessed segmentation (uint8)
- `{case_id}_metadata.json`: Processing metadata

### Create Train/Val/Test Splits
```bash
python create_splits.py
```

**Default Split Ratios**:
- Training: 70% (253 cases)
- Validation: 15% (54 cases)
- Testing: 15% (54 cases)

**Output**:
- `data/splits/train_cases.json`
- `data/splits/val_cases.json`
- `data/splits/test_cases.json`
- `data/splits/splits_info.json`

## Data Augmentation

### Training Augmentation
Applied on-the-fly during training:

1. **Geometric Transforms**:
   - Random rotation: ±15° (p=0.5)
   - Horizontal flip (p=0.5)
   - Vertical flip (p=0.3)
   - Elastic deformation: α=50, σ=5 (p=0.3)

2. **Intensity Transforms**:
   - Random brightness/contrast: ±20% (p=0.5)
   - Gaussian noise: var ∈ [0.001, 0.005] (p=0.3)

3. **Normalization**:
   - Mean: 0.5, Std: 0.5
   - Output range: approximately [-1, 1]

### Validation/Test Augmentation
- **Only normalization** (mean=0.5, std=0.5)
- No geometric or intensity augmentation

## Quality Checks

### Expected Statistics (after preprocessing)
- **CT intensity**: [0, 1] (before augmentation normalization)
- **Segmentation labels**: {0, 1, 2, 3}
- **Typical volume shape**: Variable (depends on resampling)
- **Organ presence**: All 3 organs should be present in most cases

### Validation
Check `data/processed/processing_summary.json` for:
- Number of successfully processed cases
- Failed cases (if any)
- Processing statistics

### Visual Inspection
Use `notebooks/01_data_exploration.ipynb` to:
- Visualize preprocessed slices
- Check organ label distribution
- Verify intensity distributions
- Inspect augmentation effects

## References

1. Hounsfield Unit Windowing:
   - Soft tissue window: 40/400 is standard for abdominal imaging
   - Reference: Radiological Society of North America (RSNA)

2. Resampling:
   - Isotropic spacing improves 3D CNN performance
   - Reference: Çiçek et al., 3D U-Net, MICCAI 2016

3. Data Augmentation:
   - Elastic deformation effective for medical images
   - Reference: Ronneberger et al., U-Net, MICCAI 2015
