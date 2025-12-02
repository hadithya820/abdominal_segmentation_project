#!/usr/bin/env python3
"""
Task 3: Comprehensive Evaluation Runner

This script runs complete evaluation for trained segmentation models including:
- Quantitative metrics (Dice, IoU, Pixel Accuracy, Hausdorff Distance)
- Statistical testing
- Visualization generation
- Error analysis
- Comparison reports

IMPORTANT: Uses data splits from data/splits/ folder for fair comparison.

Usage:
    # Evaluate single model (U-Net baseline)
    python run_evaluation.py --model unet --checkpoint results/unet_baseline/checkpoints/best_model.pth
    
    # Evaluate multiple models and compare
    python run_evaluation.py --compare --models unet attention_unet --checkpoints path1 path2
    
    # Run on test set
    python run_evaluation.py --model unet --split test --checkpoint path/to/checkpoint.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Add paths
sys.path.append(str(Path(__file__).parent))

from models.unet_2d import UNet2D
from models.attention_unet_2d import AttentionUNet2D
from utils.dataset import AbdomenCTDataset, get_validation_augmentation
from metrics import SegmentationMetrics, BatchMetricsCalculator, StatisticalTests, format_metrics_table
from error_analysis import ErrorAnalyzer
from visualization.visualization_suite import SegmentationVisualizer, MetricsVisualizer


# Model registry - add new models here
MODEL_REGISTRY = {
    'unet': {
        'class': UNet2D,
        'config': 'config/unet_train_config.json',
        'name': 'U-Net 2D'
    },
    # Future models can be added here:
    'attention_unet': {
        'class': AttentionUNet2D,
        'config': 'config/attention_unet_train_config.json',
        'name': 'Attention U-Net'
    },
    # 'unet3d': {
    #     'class': UNet3D,
    #     'config': 'config/unet3d_config.json',
    #     'name': '3D U-Net'
    # },
}


class ComprehensiveEvaluationRunner:
    """
    Runs comprehensive evaluation pipeline for segmentation models.
    """
    
    def __init__(self, output_dir: str = 'results/evaluation'):
        """
        Initialize evaluation runner.
        
        Args:
            output_dir: Directory for all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.visualizer = SegmentationVisualizer(
            class_names=['Background', 'Liver', 'Kidneys', 'Spleen']
        )
        self.metrics_visualizer = MetricsVisualizer()
        
        # Results storage
        self.all_results = {}
        self.all_raw_scores = {}
    
    def load_model(self, model_type: str, config_path: str, 
                   checkpoint_path: str) -> nn.Module:
        """
        Load a model from checkpoint.
        
        Args:
            model_type: Type of model ('unet', 'attention_unet', etc.)
            config_path: Path to config JSON
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get model class
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
        
        model_info = MODEL_REGISTRY[model_type]
        model_class = model_info['class']
        
        # Create model
        model = model_class(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config.get('bilinear', False)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded {model_info['name']} from epoch {epoch}")
        
        return model, config
    
    def create_dataloader(self, config: dict, split: str = 'val') -> DataLoader:
        """
        Create data loader for specified split.
        
        IMPORTANT: Uses standard splits from data/splits/ for fair comparison.
        
        Args:
            config: Model configuration
            split: Data split ('train', 'val', 'test')
            
        Returns:
            DataLoader for the split
        """
        # Use standard split files
        split_files = {
            'train': 'data/splits/train_cases.json',
            'val': 'data/splits/val_cases.json',
            'test': 'data/splits/test_cases.json'
        }
        
        split_file = split_files.get(split)
        if not split_file:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Using split file: {split_file}")
        
        dataset = AbdomenCTDataset(
            data_dir=config['data_dir'],
            split_file=split_file,
            transform=get_validation_augmentation(),
            cache_data=False
        )
        
        print(f"Loaded {len(dataset)} samples from {split} set")
        
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        return loader, dataset
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader,
                       dataset: AbdomenCTDataset,
                       model_name: str,
                       compute_hausdorff: bool = True,
                       generate_visualizations: bool = True,
                       n_vis_samples: int = 8) -> dict:
        """
        Run comprehensive evaluation on a model.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
            dataset: Dataset object (for visualization)
            model_name: Name identifier for the model
            compute_hausdorff: Whether to compute Hausdorff distance
            generate_visualizations: Whether to generate visualization images
            n_vis_samples: Number of samples to visualize
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Initialize calculators
        metrics_calc = BatchMetricsCalculator(
            num_classes=4,
            class_names=['Liver', 'Kidneys', 'Spleen']
        )
        error_analyzer = ErrorAnalyzer(
            num_classes=4,
            class_names=['Liver', 'Kidneys', 'Spleen']
        )
        
        # Storage for visualizations
        vis_samples = []
        all_predictions = []
        all_targets = []
        sample_idx = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(data_loader, 
                                                              desc="Evaluating")):
                images = images.to(self.device)
                masks_np = masks.numpy()
                
                # Forward pass
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Process each sample
                for pred, target in zip(predictions, masks_np):
                    # Update metrics
                    metrics_calc.update(pred, target, compute_hausdorff=compute_hausdorff)
                    
                    # Update error analysis
                    error_analyzer.update(pred, target, sample_id=f'sample_{sample_idx}')
                    
                    all_predictions.append(pred)
                    all_targets.append(target)
                    sample_idx += 1
                
                # Collect visualization samples
                if generate_visualizations and len(vis_samples) < n_vis_samples:
                    for i in range(min(len(predictions), n_vis_samples - len(vis_samples))):
                        img_np = images[i].cpu().numpy().squeeze()
                        vis_samples.append((img_np, masks_np[i], predictions[i]))
        
        # Get results
        summary = metrics_calc.compute_summary()
        raw_scores = metrics_calc.get_raw_scores()
        
        # Generate outputs
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save error analysis
        error_analyzer.save_results(model_output_dir / 'error_analysis')
        
        # Generate visualizations
        if generate_visualizations and vis_samples:
            vis_dir = model_output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            # Grid visualization
            self.visualizer.visualize_grid(
                vis_samples,
                save_path=str(vis_dir / 'prediction_grid.png'),
                show=False
            )
            
            # Individual samples with error maps
            for i, (img, gt, pred) in enumerate(vis_samples[:4]):
                self.visualizer.visualize_single_prediction(
                    img, gt, pred,
                    title=f'{model_name} - Sample {i+1}',
                    save_path=str(vis_dir / f'sample_{i+1}.png'),
                    show=False
                )
                self.visualizer.visualize_error_map(
                    gt, pred, img,
                    save_path=str(vis_dir / f'error_map_{i+1}.png'),
                    show=False
                )
        
        # Print results
        print(f"\n{format_metrics_table(summary, 'dice')}")
        print(f"\n{format_metrics_table(summary, 'iou')}")
        
        # Save results
        results = {
            'model_name': model_name,
            'n_samples': sample_idx,
            'summary': summary,
            'error_report': error_analyzer.generate_report()
        }
        
        # Save summary JSON
        self._save_summary_json(results, model_output_dir / 'evaluation_summary.json')
        
        # Store for comparison
        self.all_results[model_name] = results
        self.all_raw_scores[model_name] = raw_scores
        
        return results
    
    def compare_models(self, model_names: list = None):
        """
        Generate comparison between evaluated models.
        
        Args:
            model_names: List of model names to compare (None = all)
        """
        if model_names is None:
            model_names = list(self.all_results.keys())
        
        if len(model_names) < 2:
            print("Need at least 2 models for comparison")
            return
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        comparison_dir = self.output_dir / 'comparison'
        comparison_dir.mkdir(exist_ok=True)
        
        # Generate comparison tables
        comparison_report = self._generate_comparison_report(model_names)
        
        with open(comparison_dir / 'comparison_report.txt', 'w') as f:
            f.write(comparison_report)
        
        print(comparison_report)
        
        # Statistical tests between consecutive models
        for i in range(len(model_names) - 1):
            m1, m2 = model_names[i], model_names[i+1]
            print(f"\nStatistical comparison: {m1} vs {m2}")
            
            scores1 = self.all_raw_scores[m1]['dice']
            scores2 = self.all_raw_scores[m2]['dice']
            
            for class_name in ['Liver', 'Kidneys', 'Spleen']:
                s1 = scores1.get(class_name, [])
                s2 = scores2.get(class_name, [])
                
                if len(s1) == len(s2) and len(s1) > 0:
                    t_test = StatisticalTests.paired_t_test(s1, s2)
                    wilcoxon = StatisticalTests.wilcoxon_test(s1, s2)
                    
                    sig = "***" if t_test['p_value'] < 0.001 else "**" if t_test['p_value'] < 0.01 else "*" if t_test['p_value'] < 0.05 else ""
                    print(f"  {class_name}: diff = {t_test['mean_diff']:.4f}, p = {t_test['p_value']:.4f} {sig}")
        
        # Generate comparison visualizations
        if len(model_names) >= 2:
            # Dice distribution comparison
            dice_scores = {name: self.all_raw_scores[name]['dice'] 
                          for name in model_names}
            
            self.metrics_visualizer.plot_dice_distribution(
                dice_scores,
                save_path=str(comparison_dir / 'dice_distribution.png'),
                show=False
            )
            
            # Metrics comparison bar chart
            summary_dict = {name: self.all_results[name]['summary'] 
                          for name in model_names}
            
            self.metrics_visualizer.plot_metrics_comparison(
                summary_dict,
                metric='dice',
                save_path=str(comparison_dir / 'dice_comparison.png'),
                show=False
            )
            
            self.metrics_visualizer.plot_metrics_comparison(
                summary_dict,
                metric='iou',
                save_path=str(comparison_dir / 'iou_comparison.png'),
                show=False
            )
        
        # Generate LaTeX tables
        self._generate_latex_tables(model_names, comparison_dir)
        
        print(f"\nComparison results saved to {comparison_dir}")
    
    def _generate_comparison_report(self, model_names: list) -> str:
        """Generate formatted comparison report."""
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        # Dice comparison table
        lines.append("\nDICE SCORE COMPARISON")
        lines.append("-" * 80)
        
        header = f"{'Organ':<15}"
        for name in model_names:
            header += f" {name:>20}"
        lines.append(header)
        lines.append("-" * 80)
        
        for class_name in ['Liver', 'Kidneys', 'Spleen']:
            row = f"{class_name:<15}"
            for name in model_names:
                summary = self.all_results[name]['summary']['dice']
                if class_name in summary:
                    mean = summary[class_name]['mean']
                    std = summary[class_name]['std']
                    row += f" {mean:.4f}±{std:.4f}".rjust(20)
            lines.append(row)
        
        lines.append("-" * 80)
        row = f"{'OVERALL':<15}"
        for name in model_names:
            summary = self.all_results[name]['summary']['dice']
            if 'mean' in summary:
                mean = summary['mean']['mean']
                row += f" {mean:.4f}".rjust(20)
        lines.append(row)
        
        # IoU comparison
        lines.append("\n\nIoU SCORE COMPARISON")
        lines.append("-" * 80)
        lines.append(header)
        lines.append("-" * 80)
        
        for class_name in ['Liver', 'Kidneys', 'Spleen']:
            row = f"{class_name:<15}"
            for name in model_names:
                summary = self.all_results[name]['summary']['iou']
                if class_name in summary:
                    mean = summary[class_name]['mean']
                    std = summary[class_name]['std']
                    row += f" {mean:.4f}±{std:.4f}".rjust(20)
            lines.append(row)
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def _generate_latex_tables(self, model_names: list, output_dir: Path):
        """Generate LaTeX formatted tables."""
        lines = []
        
        # Dice table
        lines.append("% Dice Score Comparison")
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{l" + "c" * len(model_names) + "}")
        lines.append("\\toprule")
        
        header = "\\textbf{Organ}"
        for name in model_names:
            header += f" & \\textbf{{{name}}}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        for class_name in ['Liver', 'Kidneys', 'Spleen']:
            row = class_name
            for name in model_names:
                summary = self.all_results[name]['summary']['dice']
                if class_name in summary:
                    mean = summary[class_name]['mean']
                    std = summary[class_name]['std']
                    row += f" & ${mean:.4f} \\pm {std:.4f}$"
            row += " \\\\"
            lines.append(row)
        
        lines.append("\\midrule")
        row = "\\textbf{Overall}"
        for name in model_names:
            summary = self.all_results[name]['summary']['dice']
            if 'mean' in summary:
                mean = summary['mean']['mean']
                row += f" & \\textbf{{{mean:.4f}}}"
        row += " \\\\"
        lines.append(row)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Dice coefficient comparison across models}")
        lines.append("\\label{tab:dice_comparison}")
        lines.append("\\end{table}")
        
        with open(output_dir / 'latex_tables.tex', 'w') as f:
            f.write('\n'.join(lines))
    
    def _save_summary_json(self, results: dict, filepath: Path):
        """Save results summary to JSON."""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        results_clean = convert_numpy(results)
        
        # Remove error report from JSON (save separately)
        if 'error_report' in results_clean:
            del results_clean['error_report']
        
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Evaluation for Multi-Organ Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate U-Net baseline on validation set
  python run_evaluation.py --model unet --checkpoint results/unet_baseline/checkpoints/best_model.pth
  
  # Evaluate on test set
  python run_evaluation.py --model unet --checkpoint path/to/checkpoint.pth --split test
  
  # Skip Hausdorff distance for faster evaluation
  python run_evaluation.py --model unet --checkpoint path/to/checkpoint.pth --no_hausdorff
  
  # Compare multiple models
  python run_evaluation.py --compare --models unet attention_unet --checkpoints ckpt1.pth ckpt2.pth
        """
    )
    
    parser.add_argument('--model', type=str, default='unet',
                       help='Model type to evaluate (default: unet)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON (default: use model default)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Data split to evaluate (default: val)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory (default: results/evaluation)')
    parser.add_argument('--no_hausdorff', action='store_true',
                       help='Skip Hausdorff distance computation')
    parser.add_argument('--no_vis', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--n_vis', type=int, default=8,
                       help='Number of samples to visualize (default: 8)')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    parser.add_argument('--models', nargs='+', type=str,
                       help='Model types for comparison')
    parser.add_argument('--checkpoints', nargs='+', type=str,
                       help='Checkpoint paths for comparison')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ComprehensiveEvaluationRunner(args.output_dir)
    
    if args.compare:
        # Multi-model comparison mode
        if not args.models or not args.checkpoints:
            parser.error("--compare requires --models and --checkpoints")
        if len(args.models) != len(args.checkpoints):
            parser.error("Number of models must match number of checkpoints")
        
        for model_type, ckpt in zip(args.models, args.checkpoints):
            config_path = args.config or MODEL_REGISTRY[model_type]['config']
            model, config = runner.load_model(model_type, config_path, ckpt)
            loader, dataset = runner.create_dataloader(config, args.split)
            
            runner.evaluate_model(
                model, loader, dataset,
                model_name=MODEL_REGISTRY[model_type]['name'],
                compute_hausdorff=not args.no_hausdorff,
                generate_visualizations=not args.no_vis,
                n_vis_samples=args.n_vis
            )
        
        runner.compare_models()
    
    else:
        # Single model evaluation
        config_path = args.config or MODEL_REGISTRY[args.model]['config']
        model, config = runner.load_model(args.model, config_path, args.checkpoint)
        loader, dataset = runner.create_dataloader(config, args.split)
        
        runner.evaluate_model(
            model, loader, dataset,
            model_name=MODEL_REGISTRY[args.model]['name'],
            compute_hausdorff=not args.no_hausdorff,
            generate_visualizations=not args.no_vis,
            n_vis_samples=args.n_vis
        )
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
