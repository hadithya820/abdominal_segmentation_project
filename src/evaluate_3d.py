"""
Evaluation Script for 3D Segmentation Models

Features:
- Load trained checkpoints
- Inference on full volumes with sliding window
- Compute comprehensive metrics (Dice, IoU, HD95, ASSD)
- Generate visualizations
- Compare multiple models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.unet_3d import UNet3D
from models.vnet import VNet
from models.segresnet import SegResNet


def get_model(model_name: str, n_channels: int = 1, n_classes: int = 4, 
              base_filters: int = 32, deep_supervision: bool = False) -> nn.Module:
    """Create model by name"""
    model_name = model_name.lower()
    
    if model_name == 'unet3d':
        return UNet3D(n_channels, n_classes, base_filters, use_checkpoint=False, deep_supervision=deep_supervision)
    elif model_name == 'vnet':
        return VNet(n_channels, n_classes, base_filters // 2, use_checkpoint=False, deep_supervision=deep_supervision)
    elif model_name == 'segresnet':
        return SegResNet(n_channels, n_classes, base_filters, use_checkpoint=False, deep_supervision=deep_supervision)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = get_model(
        config['model_name'],
        config.get('n_channels', 1),
        config.get('n_classes', 4),
        config.get('base_filters', 32),
        config.get('deep_supervision', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def sliding_window_inference(
    model: nn.Module,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    overlap: float = 0.5,
    batch_size: int = 4,
    device: torch.device = torch.device('cuda'),
    use_gaussian: bool = True
) -> np.ndarray:
    """
    Perform inference on a full volume using sliding window approach
    
    Args:
        model: Trained segmentation model
        volume: Input volume (D, H, W)
        patch_size: Size of each patch
        overlap: Overlap ratio between patches
        batch_size: Number of patches to process at once
        device: Device to run inference on
        use_gaussian: Use Gaussian weighting for overlapping regions
    
    Returns:
        Predicted segmentation (D, H, W)
    """
    model.eval()
    
    # Get volume shape
    vol_shape = volume.shape
    n_classes = 4  # Assuming 4 classes
    
    # Calculate step size
    step = [int(p * (1 - overlap)) for p in patch_size]
    
    # Initialize output arrays
    output = np.zeros((n_classes,) + vol_shape, dtype=np.float32)
    count = np.zeros(vol_shape, dtype=np.float32)
    
    # Create Gaussian importance map for patch weighting
    if use_gaussian:
        sigma = [p / 4 for p in patch_size]
        importance_map = _create_gaussian_importance_map(patch_size, sigma)
    else:
        importance_map = np.ones(patch_size, dtype=np.float32)
    
    # Generate patch coordinates
    coords = []
    for z in range(0, max(1, vol_shape[0] - patch_size[0] + 1), step[0]):
        for y in range(0, max(1, vol_shape[1] - patch_size[1] + 1), step[1]):
            for x in range(0, max(1, vol_shape[2] - patch_size[2] + 1), step[2]):
                coords.append((z, y, x))
    
    # Add edge coordinates to ensure full coverage
    if vol_shape[0] > patch_size[0]:
        coords.append((vol_shape[0] - patch_size[0], 0, 0))
    if vol_shape[1] > patch_size[1]:
        coords.append((0, vol_shape[1] - patch_size[1], 0))
    if vol_shape[2] > patch_size[2]:
        coords.append((0, 0, vol_shape[2] - patch_size[2]))
    
    # Process patches in batches
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            batch_patches = []
            
            for z, y, x in batch_coords:
                patch = volume[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                
                # Pad if necessary
                if patch.shape != tuple(patch_size):
                    padded = np.zeros(patch_size, dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                
                batch_patches.append(patch)
            
            # Stack and convert to tensor
            batch_tensor = torch.tensor(np.stack(batch_patches), dtype=torch.float32)
            batch_tensor = batch_tensor.unsqueeze(1).to(device)  # (B, 1, D, H, W)
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                pred = model(batch_tensor)
                if isinstance(pred, dict):
                    pred = pred['logits']
                elif isinstance(pred, (list, tuple)):
                    pred = pred[0]
            
            pred = F.softmax(pred, dim=1).cpu().numpy()
            
            # Add predictions to output
            for i, (z, y, x) in enumerate(batch_coords):
                z_end = min(z + patch_size[0], vol_shape[0])
                y_end = min(y + patch_size[1], vol_shape[1])
                x_end = min(x + patch_size[2], vol_shape[2])
                
                dz, dy, dx = z_end - z, y_end - y, x_end - x
                
                output[:, z:z_end, y:y_end, x:x_end] += pred[i, :, :dz, :dy, :dx] * importance_map[:dz, :dy, :dx]
                count[z:z_end, y:y_end, x:x_end] += importance_map[:dz, :dy, :dx]
    
    # Average overlapping regions
    output = output / np.maximum(count, 1e-8)
    
    # Get final predictions
    return np.argmax(output, axis=0).astype(np.uint8)


def _create_gaussian_importance_map(shape: Tuple[int, int, int], sigma: List[float]) -> np.ndarray:
    """Create Gaussian importance map for weighting overlapping patches"""
    tmp = np.zeros(shape, dtype=np.float32)
    center = [s // 2 for s in shape]
    tmp[center[0], center[1], center[2]] = 1
    importance_map = ndimage.gaussian_filter(tmp, sigma=sigma, mode='constant')
    importance_map = importance_map / importance_map.max()
    importance_map = importance_map ** 0.5  # Reduce influence of Gaussian
    return importance_map


def compute_dice(pred: np.ndarray, target: np.ndarray, n_classes: int = 4) -> Dict[int, float]:
    """Compute Dice score per class"""
    dice_scores = {}
    
    for c in range(n_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c)
        
        if union > 0:
            dice_scores[c] = 2 * intersection / union
        else:
            dice_scores[c] = 1.0 if np.sum(target_c) == 0 else 0.0
    
    return dice_scores


def compute_iou(pred: np.ndarray, target: np.ndarray, n_classes: int = 4) -> Dict[int, float]:
    """Compute IoU (Jaccard) score per class"""
    iou_scores = {}
    
    for c in range(n_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c) - intersection
        
        if union > 0:
            iou_scores[c] = intersection / union
        else:
            iou_scores[c] = 1.0 if np.sum(target_c) == 0 else 0.0
    
    return iou_scores


def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: float = 95) -> float:
    """
    Compute Hausdorff Distance at given percentile (HD95 by default)
    """
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.inf
    
    # Get surface points
    pred_boundary = pred ^ ndimage.binary_erosion(pred)
    target_boundary = target ^ ndimage.binary_erosion(target)
    
    # Compute distance transforms
    dist_pred = distance_transform_edt(~target_boundary)
    dist_target = distance_transform_edt(~pred_boundary)
    
    # Get distances at boundary points
    pred_distances = dist_pred[pred_boundary > 0]
    target_distances = dist_target[target_boundary > 0]
    
    if len(pred_distances) == 0 or len(target_distances) == 0:
        return np.inf
    
    # Compute percentile Hausdorff distance
    hd = max(np.percentile(pred_distances, percentile),
             np.percentile(target_distances, percentile))
    
    return hd


def evaluate_case(
    model: nn.Module,
    ct_path: str,
    seg_path: str,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    overlap: float = 0.5,
    device: torch.device = torch.device('cuda')
) -> Dict[str, Any]:
    """Evaluate model on a single case"""
    
    # Load data
    ct_volume = np.load(ct_path).astype(np.float32)
    seg_volume = np.load(seg_path).astype(np.uint8)
    
    # Run inference
    pred = sliding_window_inference(
        model, ct_volume, 
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )
    
    # Compute metrics
    n_classes = 4
    
    dice_scores = compute_dice(pred, seg_volume, n_classes)
    iou_scores = compute_iou(pred, seg_volume, n_classes)
    
    # Compute HD95 per class (skip background)
    hd95_scores = {}
    for c in range(1, n_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (seg_volume == c).astype(np.uint8)
        
        if np.sum(pred_c) > 0 and np.sum(target_c) > 0:
            hd95_scores[c] = compute_hausdorff_distance(pred_c, target_c, percentile=95)
        else:
            hd95_scores[c] = np.inf
    
    return {
        'dice': dice_scores,
        'iou': iou_scores,
        'hd95': hd95_scores,
        'prediction': pred,
        'volume_shape': ct_volume.shape
    }


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    test_split: str,
    output_dir: str,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    overlap: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate model on test set
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to processed data directory
        test_split: Path to test split JSON
        output_dir: Path to save results
        patch_size: Patch size for inference
        overlap: Overlap ratio
    
    Returns:
        Evaluation results dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    
    # Load test cases
    with open(test_split, 'r') as f:
        test_cases = json.load(f)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each case
    all_results = []
    class_names = ['background', 'liver', 'kidneys', 'spleen']
    
    print(f"\nEvaluating {len(test_cases)} test cases...")
    
    for case_id in tqdm(test_cases):
        ct_path = data_dir / f"{case_id}_ct.npy"
        seg_path = data_dir / f"{case_id}_seg.npy"
        
        if not ct_path.exists() or not seg_path.exists():
            warnings.warn(f"Missing files for {case_id}, skipping")
            continue
        
        result = evaluate_case(
            model, str(ct_path), str(seg_path),
            patch_size=patch_size, overlap=overlap, device=device
        )
        result['case_id'] = case_id
        all_results.append(result)
    
    # Aggregate results
    aggregated = {
        'model_name': config['model_name'],
        'n_cases': len(all_results),
        'per_class_metrics': {},
        'per_case_results': []
    }
    
    n_classes = 4
    
    for c in range(n_classes):
        dice_values = [r['dice'][c] for r in all_results]
        iou_values = [r['iou'][c] for r in all_results]
        
        aggregated['per_class_metrics'][class_names[c]] = {
            'dice_mean': np.mean(dice_values),
            'dice_std': np.std(dice_values),
            'iou_mean': np.mean(iou_values),
            'iou_std': np.std(iou_values)
        }
        
        if c > 0:  # Skip background for HD95
            hd95_values = [r['hd95'][c] for r in all_results if not np.isinf(r['hd95'][c])]
            if hd95_values:
                aggregated['per_class_metrics'][class_names[c]]['hd95_mean'] = np.mean(hd95_values)
                aggregated['per_class_metrics'][class_names[c]]['hd95_std'] = np.std(hd95_values)
    
    # Overall metrics (excluding background)
    all_dice = []
    all_iou = []
    for c in range(1, n_classes):
        all_dice.extend([r['dice'][c] for r in all_results])
        all_iou.extend([r['iou'][c] for r in all_results])
    
    aggregated['overall_metrics'] = {
        'mean_dice': np.mean(all_dice),
        'mean_iou': np.mean(all_iou)
    }
    
    # Per-case results (without predictions for JSON serialization)
    for r in all_results:
        case_result = {
            'case_id': r['case_id'],
            'dice': {class_names[c]: r['dice'][c] for c in range(n_classes)},
            'iou': {class_names[c]: r['iou'][c] for c in range(n_classes)},
            'hd95': {class_names[c]: r['hd95'].get(c, np.inf) for c in range(1, n_classes)}
        }
        aggregated['per_case_results'].append(case_result)
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    
    # Convert numpy types to Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(aggregated), f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {config['model_name']}")
    print("="*60)
    
    print(f"\nOverall Mean Dice: {aggregated['overall_metrics']['mean_dice']:.4f}")
    print(f"Overall Mean IoU: {aggregated['overall_metrics']['mean_iou']:.4f}")
    
    print("\nPer-Class Results:")
    print("-"*60)
    print(f"{'Class':<12} {'Dice':>12} {'IoU':>12} {'HD95':>12}")
    print("-"*60)
    
    for class_name in class_names[1:]:  # Skip background
        metrics = aggregated['per_class_metrics'][class_name]
        dice_str = f"{metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f}"
        iou_str = f"{metrics['iou_mean']:.4f}±{metrics['iou_std']:.4f}"
        hd95_str = f"{metrics.get('hd95_mean', np.inf):.2f}±{metrics.get('hd95_std', 0):.2f}" if 'hd95_mean' in metrics else "N/A"
        print(f"{class_name:<12} {dice_str:>12} {iou_str:>12} {hd95_str:>12}")
    
    print("-"*60)
    print(f"\nResults saved to: {results_path}")
    
    return aggregated


def compare_models(
    checkpoints: Dict[str, str],
    data_dir: str,
    test_split: str,
    output_dir: str
) -> Dict[str, Any]:
    """Compare multiple models"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison = {}
    
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print('='*60)
        
        model_output_dir = output_dir / model_name
        results = evaluate_model(
            checkpoint_path, data_dir, test_split, str(model_output_dir)
        )
        comparison[model_name] = results
    
    # Save comparison
    comparison_path = output_dir / 'model_comparison.json'
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<15} {'Mean Dice':>12} {'Mean IoU':>12} {'Liver':>10} {'Kidneys':>10} {'Spleen':>10}")
    print("-"*80)
    
    for model_name, results in comparison.items():
        mean_dice = results['overall_metrics']['mean_dice']
        mean_iou = results['overall_metrics']['mean_iou']
        liver_dice = results['per_class_metrics']['liver']['dice_mean']
        kidneys_dice = results['per_class_metrics']['kidneys']['dice_mean']
        spleen_dice = results['per_class_metrics']['spleen']['dice_mean']
        
        print(f"{model_name:<15} {mean_dice:>12.4f} {mean_iou:>12.4f} {liver_dice:>10.4f} {kidneys_dice:>10.4f} {spleen_dice:>10.4f}")
    
    print("-"*80)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D segmentation models')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--test_split', type=str, default='data/splits/test_cases.json', help='Test split JSON')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='Output directory')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[64, 128, 128], help='Patch size')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio')
    parser.add_argument('--compare', action='store_true', help='Compare all trained models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all trained models
        checkpoints = {
            'UNet3D': 'results/unet3d_baseline/checkpoints/best_checkpoint.pth',
            'VNet': 'results/vnet_baseline/checkpoints/best_checkpoint.pth',
            'SegResNet': 'results/segresnet_baseline/checkpoints/best_checkpoint.pth'
        }
        
        # Filter existing checkpoints
        checkpoints = {k: v for k, v in checkpoints.items() if Path(v).exists()}
        
        if not checkpoints:
            print("No trained checkpoints found!")
            return
        
        compare_models(checkpoints, args.data_dir, args.test_split, args.output_dir)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required when not using --compare")
        
        evaluate_model(
            args.checkpoint,
            args.data_dir,
            args.test_split,
            args.output_dir,
            tuple(args.patch_size),
            args.overlap
        )


if __name__ == "__main__":
    main()
