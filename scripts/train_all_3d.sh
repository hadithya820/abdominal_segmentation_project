#!/bin/bash
# Train all 3D segmentation models
# Optimized for g6.16xlarge [L4] with 32GB GPU memory

set -e

echo "=========================================="
echo "3D Model Training Pipeline"
echo "=========================================="

# Activate environment if needed
# source /path/to/venv/bin/activate

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Run profiling first
echo ""
echo "Step 1: Running memory profiling..."
echo "=========================================="
python src/profile_models.py --quick --output results

# Train UNet3D
echo ""
echo "Step 2: Training UNet3D..."
echo "=========================================="
python src/train_3d.py --config config/unet3d_config.json

# Train VNet
echo ""
echo "Step 3: Training VNet..."
echo "=========================================="
python src/train_3d.py --config config/vnet_config.json

# Train SegResNet
echo ""
echo "Step 4: Training SegResNet..."
echo "=========================================="
python src/train_3d.py --config config/segresnet_config.json

echo ""
echo "=========================================="
echo "All training complete!"
echo "=========================================="
echo "Results saved in:"
echo "  - results/unet3d_baseline/"
echo "  - results/vnet_baseline/"
echo "  - results/segresnet_baseline/"
echo ""
