# Train all 3D segmentation models
# Optimized for g6.16xlarge [L4] with 32GB GPU memory
# Run from project root directory

Write-Host "=========================================="
Write-Host "3D Model Training Pipeline"
Write-Host "=========================================="

$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

# Run profiling first
Write-Host ""
Write-Host "Step 1: Running memory profiling..."
Write-Host "=========================================="
python src/profile_models.py --quick --output results

# Train UNet3D
Write-Host ""
Write-Host "Step 2: Training UNet3D..."
Write-Host "=========================================="
python src/train_3d.py --config config/unet3d_config.json

# Train VNet
Write-Host ""
Write-Host "Step 3: Training VNet..."
Write-Host "=========================================="
python src/train_3d.py --config config/vnet_config.json

# Train SegResNet
Write-Host ""
Write-Host "Step 4: Training SegResNet..."
Write-Host "=========================================="
python src/train_3d.py --config config/segresnet_config.json

Write-Host ""
Write-Host "=========================================="
Write-Host "All training complete!"
Write-Host "=========================================="
Write-Host "Results saved in:"
Write-Host "  - results/unet3d_baseline/"
Write-Host "  - results/vnet_baseline/"
Write-Host "  - results/segresnet_baseline/"
Write-Host ""
