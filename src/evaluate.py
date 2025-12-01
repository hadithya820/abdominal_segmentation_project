"""
Comprehensive Evaluation Script for Multi-Organ Segmentation Models

This script evaluates trained models on the test set and generates:
- Quantitative metrics (Dice, IoU, Pixel Accuracy, Hausdorff Distance)
- Statistical comparisons between models
- Comparison tables and figures
- Error analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.unet_2d import UNet2D
from utils.dataset import AbdomenCTDataset, get_validation_augmentation
from metrics import SegmentationMetrics, BatchMetricsCalculator, StatisticalTests, format_metrics_table


class ModelEvaluator:
    """
    Evaluates segmentation models and computes comprehensive metrics.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, 
                 model_name: str = 'UNet2D', device: str = None):
        """
        Initialize evaluator with model configuration.
        
        Args:
            config_path: Path to training configuration JSON
            checkpoint_path: Path to model checkpoint
            model_name: Name identifier for the model
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Initialize metrics calculator
        self.metrics_calculator = BatchMetricsCalculator(
            num_classes=self.config['n_classes'],
            class_names=['Liver', 'Kidneys', 'Spleen']
        )
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        print(f"Loading {self.model_name} from {checkpoint_path}...")
        
        # Create model architecture
        model = UNet2D(
            n_channels=self.config['n_channels'],
            n_classes=self.config['n_classes'],
            bilinear=self.config.get('bilinear', False)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {self.epoch}")
        
        return model
    
    def evaluate(self, data_loader: DataLoader, 
                 compute_hausdorff: bool = True,
                 save_predictions: bool = False,
                 output_dir: Path = None) -> dict:
        """
        Evaluate model on given data loader.
        
        Args:
            data_loader: DataLoader for evaluation data
            compute_hausdorff: Whether to compute Hausdorff distance
            save_predictions: Whether to save predictions
            output_dir: Directory to save predictions
            
        Returns:
            Dictionary with evaluation results
        """
        self.metrics_calculator.reset()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(data_loader, 
                                                              desc=f"Evaluating {self.model_name}")):
                images = images.to(self.device)
                masks = masks.numpy()
                
                # Forward pass
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Compute metrics for each sample in batch
                for pred, target in zip(predictions, masks):
                    self.metrics_calculator.update(
                        pred, target, 
                        compute_hausdorff=compute_hausdorff
                    )
                    
                    if save_predictions:
                        all_predictions.append(pred)
                        all_targets.append(target)
        
        # Get summary
        summary = self.metrics_calculator.compute_summary()
        raw_scores = self.metrics_calculator.get_raw_scores()
        
        # Save predictions if requested
        if save_predictions and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / f'{self.model_name}_predictions.npy', 
                   np.array(all_predictions))
            np.save(output_dir / f'{self.model_name}_targets.npy', 
                   np.array(all_targets))
        
        return {
            'model_name': self.model_name,
            'epoch': self.epoch,
            'summary': summary,
            'raw_scores': raw_scores
        }


class ComprehensiveEvaluator:
    """
    Evaluates multiple models and generates comparison reports.
    """
    
    def __init__(self, output_dir: str = 'results/evaluation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def add_model_results(self, results: dict):
        """Add evaluation results for a model."""
        self.results[results['model_name']] = results
    
    def compare_models(self, model1_name: str, model2_name: str, 
                       metric: str = 'dice') -> dict:
        """
        Perform statistical comparison between two models.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            metric: Metric to compare ('dice', 'iou', etc.)
            
        Returns:
            Dictionary with comparison results
        """
        if model1_name not in self.results or model2_name not in self.results:
            raise ValueError(f"Results not found for one or both models")
        
        scores1 = self.results[model1_name]['raw_scores'][metric]
        scores2 = self.results[model2_name]['raw_scores'][metric]
        
        comparison = {}
        
        for class_name in scores1.keys():
            s1 = scores1[class_name]
            s2 = scores2[class_name]
            
            if len(s1) != len(s2):
                print(f"Warning: Different number of samples for {class_name}")
                continue
            
            comparison[class_name] = {
                'paired_t_test': StatisticalTests.paired_t_test(s1, s2),
                'wilcoxon_test': StatisticalTests.wilcoxon_test(s1, s2),
                'mean_diff': np.mean(s1) - np.mean(s2)
            }
        
        return comparison
    
    def generate_comparison_table(self, metric: str = 'dice') -> str:
        """
        Generate formatted comparison table for all models.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Formatted table string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"MODEL COMPARISON - {metric.upper()}")
        lines.append("=" * 80)
        
        # Header
        model_names = list(self.results.keys())
        header = f"{'Organ':<15}"
        for name in model_names:
            header += f" {name:>15}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Get class names from first model
        first_model = list(self.results.values())[0]
        class_names = list(first_model['summary'][metric].keys())
        
        for class_name in class_names:
            if class_name == 'mean':
                lines.append("-" * 80)
                row = f"{'OVERALL':<15}"
            else:
                row = f"{class_name:<15}"
            
            for model_name in model_names:
                summary = self.results[model_name]['summary'][metric]
                if class_name in summary:
                    mean = summary[class_name]['mean']
                    std = summary[class_name].get('std', 0)
                    row += f" {mean:.4f}±{std:.4f}"
                else:
                    row += f" {'N/A':>15}"
            
            lines.append(row)
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def generate_full_report(self) -> str:
        """Generate comprehensive evaluation report."""
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE SEGMENTATION EVALUATION REPORT")
        lines.append(f"Generated: {timestamp}")
        lines.append("=" * 80)
        lines.append("")
        
        # Per-model detailed results
        for model_name, results in self.results.items():
            lines.append(f"\n{'='*40}")
            lines.append(f"MODEL: {model_name}")
            lines.append(f"Checkpoint Epoch: {results['epoch']}")
            lines.append(f"{'='*40}\n")
            
            # Dice scores
            lines.append(format_metrics_table(results['summary'], 'dice'))
            lines.append("")
            
            # IoU scores
            lines.append(format_metrics_table(results['summary'], 'iou'))
            lines.append("")
            
            # Pixel Accuracy
            lines.append("-" * 40)
            lines.append("PIXEL ACCURACY")
            lines.append("-" * 40)
            acc = results['summary']['pixel_accuracy']
            lines.append(f"Overall: {acc['overall']['mean']:.4f} ± {acc['overall']['std']:.4f}")
            for class_name in ['Liver', 'Kidneys', 'Spleen']:
                if class_name in acc:
                    lines.append(f"{class_name}: {acc[class_name]['mean']:.4f} ± {acc[class_name]['std']:.4f}")
            lines.append("")
            
            # Hausdorff Distance
            if results['summary'].get('hausdorff_distance'):
                lines.append("-" * 40)
                lines.append("HAUSDORFF DISTANCE (95th percentile)")
                lines.append("-" * 40)
                hd = results['summary']['hausdorff_distance']
                for class_name, scores in hd.items():
                    if 'mean' in scores:
                        lines.append(f"{class_name}: {scores['mean']:.2f} ± {scores['std']:.2f} mm")
        
        # Model comparison table
        if len(self.results) > 1:
            lines.append("\n" + "=" * 80)
            lines.append("MODEL COMPARISON")
            lines.append("=" * 80 + "\n")
            lines.append(self.generate_comparison_table('dice'))
            lines.append("")
            lines.append(self.generate_comparison_table('iou'))
        
        return '\n'.join(lines)
    
    def save_report(self, filename: str = 'evaluation_report.txt'):
        """Save evaluation report to file."""
        report = self.generate_full_report()
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        
        # Also save raw results as JSON
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {
                'epoch': results['epoch'],
                'summary': results['summary']
            }
        
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=float)
        
        print(f"JSON results saved to {json_path}")
    
    def save_for_latex(self, filename: str = 'latex_tables.tex'):
        """Generate LaTeX formatted tables for the report."""
        lines = []
        
        # Dice comparison table
        lines.append("% Dice Score Comparison Table")
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        
        model_names = list(self.results.keys())
        num_models = len(model_names)
        
        lines.append("\\begin{tabular}{l" + "c" * num_models + "}")
        lines.append("\\toprule")
        
        # Header
        header = "\\textbf{Organ}"
        for name in model_names:
            header += f" & \\textbf{{{name}}}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # Data rows
        first_model = list(self.results.values())[0]
        for class_name in ['Liver', 'Kidneys', 'Spleen']:
            row = class_name
            for model_name in model_names:
                summary = self.results[model_name]['summary']['dice']
                if class_name in summary:
                    mean = summary[class_name]['mean']
                    std = summary[class_name]['std']
                    row += f" & ${mean:.4f} \\pm {std:.4f}$"
            row += " \\\\"
            lines.append(row)
        
        lines.append("\\midrule")
        
        # Overall row
        row = "\\textbf{Overall}"
        for model_name in model_names:
            summary = self.results[model_name]['summary']['dice']
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
        
        # Save
        latex_path = self.output_dir / filename
        with open(latex_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"LaTeX tables saved to {latex_path}")


def evaluate_single_model(config_path: str, checkpoint_path: str, 
                          split: str = 'test', model_name: str = 'UNet2D',
                          compute_hausdorff: bool = True) -> dict:
    """
    Evaluate a single model on specified data split.
    
    Args:
        config_path: Path to config JSON
        checkpoint_path: Path to checkpoint
        split: Data split to evaluate ('val' or 'test')
        model_name: Model name identifier
        compute_hausdorff: Whether to compute Hausdorff distance
        
    Returns:
        Evaluation results dictionary
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine split file
    if split == 'test':
        split_file = config.get('test_split', 'data/splits/test_cases.json')
    else:
        split_file = config['val_split']
    
    # Create dataset
    dataset = AbdomenCTDataset(
        data_dir=config['data_dir'],
        split_file=split_file,
        transform=get_validation_augmentation(),
        cache_data=False
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} samples from {split} set")
    
    # Create evaluator and evaluate
    evaluator = ModelEvaluator(config_path, checkpoint_path, model_name)
    results = evaluator.evaluate(data_loader, compute_hausdorff=compute_hausdorff)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config JSON')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='UNet2D',
                       help='Name identifier for the model')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Data split to evaluate')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--no_hausdorff', action='store_true',
                       help='Skip Hausdorff distance computation')
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_single_model(
        args.config,
        args.checkpoint,
        split=args.split,
        model_name=args.model_name,
        compute_hausdorff=not args.no_hausdorff
    )
    
    # Create comprehensive evaluator and generate report
    comp_eval = ComprehensiveEvaluator(args.output_dir)
    comp_eval.add_model_results(results)
    
    # Print and save report
    print("\n" + comp_eval.generate_full_report())
    comp_eval.save_report(f'{args.model_name}_evaluation.txt')
    comp_eval.save_for_latex(f'{args.model_name}_latex_tables.tex')
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
