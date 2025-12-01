"""
Error Analysis Module for Segmentation Models

Provides detailed analysis of model failures including:
- Per-class error statistics
- Failure case identification
- Boundary error analysis
- Size-based error correlation
- Confusion matrix analysis
"""

import torch
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path


class ErrorAnalyzer:
    """
    Comprehensive error analysis for segmentation predictions.
    """
    
    def __init__(self, num_classes: int = 4, 
                 class_names: List[str] = None):
        """
        Initialize error analyzer.
        
        Args:
            num_classes: Total number of classes including background
            class_names: Names for organ classes (excluding background)
        """
        self.num_classes = num_classes
        self.class_names = class_names or ['Liver', 'Kidneys', 'Spleen']
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.per_sample_errors = []
        self.boundary_errors = defaultdict(list)
        self.size_vs_dice = defaultdict(list)
        self.failure_cases = []
    
    def update(self, prediction: np.ndarray, ground_truth: np.ndarray,
               sample_id: str = None, dice_threshold: float = 0.5):
        """
        Analyze errors for a single prediction.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            sample_id: Identifier for the sample
            dice_threshold: Threshold below which a prediction is considered a failure
        """
        # Update confusion matrix
        for true_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                self.confusion_matrix[true_class, pred_class] += np.sum(
                    (ground_truth == true_class) & (prediction == pred_class)
                )
        
        # Analyze per-class errors
        sample_errors = {'sample_id': sample_id}
        is_failure = False
        
        for class_idx, class_name in enumerate(self.class_names, start=1):
            gt_mask = ground_truth == class_idx
            pred_mask = prediction == class_idx
            
            gt_pixels = np.sum(gt_mask)
            pred_pixels = np.sum(pred_mask)
            
            if gt_pixels > 0 or pred_pixels > 0:
                # Compute Dice
                intersection = np.sum(gt_mask & pred_mask)
                dice = 2 * intersection / (gt_pixels + pred_pixels + 1e-8)
                
                # Compute error types
                false_positive = np.sum(pred_mask & ~gt_mask)
                false_negative = np.sum(~pred_mask & gt_mask)
                
                # Boundary error analysis
                if gt_pixels > 0:
                    boundary_error = self._compute_boundary_error(gt_mask, pred_mask)
                    self.boundary_errors[class_name].append(boundary_error)
                
                # Size vs Dice correlation
                if gt_pixels > 0:
                    self.size_vs_dice[class_name].append((gt_pixels, dice))
                
                sample_errors[class_name] = {
                    'dice': dice,
                    'gt_pixels': int(gt_pixels),
                    'pred_pixels': int(pred_pixels),
                    'false_positive': int(false_positive),
                    'false_negative': int(false_negative),
                    'fp_rate': false_positive / (pred_pixels + 1e-8),
                    'fn_rate': false_negative / (gt_pixels + 1e-8)
                }
                
                if dice < dice_threshold and gt_pixels > 100:  # Significant GT present
                    is_failure = True
            else:
                sample_errors[class_name] = {
                    'dice': 1.0,  # Both empty
                    'gt_pixels': 0,
                    'pred_pixels': 0,
                    'false_positive': 0,
                    'false_negative': 0
                }
        
        self.per_sample_errors.append(sample_errors)
        
        if is_failure:
            self.failure_cases.append(sample_errors)
    
    def _compute_boundary_error(self, gt_mask: np.ndarray, 
                                 pred_mask: np.ndarray,
                                 boundary_width: int = 3) -> Dict:
        """
        Compute errors specifically at organ boundaries.
        
        Args:
            gt_mask: Ground truth binary mask
            pred_mask: Predicted binary mask
            boundary_width: Width of boundary region to analyze
            
        Returns:
            Dictionary with boundary error statistics
        """
        # Get boundary regions
        struct = ndimage.generate_binary_structure(2, 1)
        
        gt_eroded = ndimage.binary_erosion(gt_mask, struct, iterations=boundary_width)
        gt_boundary = gt_mask & ~gt_eroded
        
        pred_eroded = ndimage.binary_erosion(pred_mask, struct, iterations=boundary_width)
        pred_boundary = pred_mask & ~pred_eroded
        
        # Compute boundary-specific metrics
        boundary_gt_pixels = np.sum(gt_boundary)
        boundary_pred_pixels = np.sum(pred_boundary)
        
        if boundary_gt_pixels > 0:
            boundary_intersection = np.sum(gt_boundary & pred_boundary)
            boundary_dice = 2 * boundary_intersection / (boundary_gt_pixels + boundary_pred_pixels + 1e-8)
            
            # Error at GT boundary
            boundary_fn = np.sum(gt_boundary & ~pred_mask)
            boundary_fn_rate = boundary_fn / boundary_gt_pixels
        else:
            boundary_dice = 1.0
            boundary_fn_rate = 0.0
        
        return {
            'boundary_dice': boundary_dice,
            'boundary_fn_rate': boundary_fn_rate,
            'boundary_pixels': int(boundary_gt_pixels)
        }
    
    def get_confusion_matrix(self, normalize: bool = True) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            normalize: Whether to normalize rows (true classes)
            
        Returns:
            Confusion matrix array
        """
        if normalize:
            row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
            return self.confusion_matrix / (row_sums + 1e-8)
        return self.confusion_matrix
    
    def get_class_statistics(self) -> Dict[str, Dict]:
        """
        Get comprehensive statistics per class.
        
        Returns:
            Dictionary with per-class statistics
        """
        stats = {}
        
        for class_name in self.class_names:
            class_errors = [s[class_name] for s in self.per_sample_errors 
                          if class_name in s]
            
            if not class_errors:
                continue
            
            dice_scores = [e['dice'] for e in class_errors]
            fp_rates = [e['fp_rate'] for e in class_errors if e['gt_pixels'] > 0 or e['pred_pixels'] > 0]
            fn_rates = [e['fn_rate'] for e in class_errors if e['gt_pixels'] > 0]
            
            stats[class_name] = {
                'dice': {
                    'mean': np.mean(dice_scores),
                    'std': np.std(dice_scores),
                    'median': np.median(dice_scores),
                    'min': np.min(dice_scores),
                    'max': np.max(dice_scores),
                    'q25': np.percentile(dice_scores, 25),
                    'q75': np.percentile(dice_scores, 75)
                },
                'false_positive_rate': {
                    'mean': np.mean(fp_rates) if fp_rates else 0,
                    'std': np.std(fp_rates) if fp_rates else 0
                },
                'false_negative_rate': {
                    'mean': np.mean(fn_rates) if fn_rates else 0,
                    'std': np.std(fn_rates) if fn_rates else 0
                },
                'n_samples': len(class_errors)
            }
            
            # Boundary statistics
            if self.boundary_errors[class_name]:
                boundary_dice = [e['boundary_dice'] for e in self.boundary_errors[class_name]]
                boundary_fn = [e['boundary_fn_rate'] for e in self.boundary_errors[class_name]]
                stats[class_name]['boundary'] = {
                    'dice_mean': np.mean(boundary_dice),
                    'dice_std': np.std(boundary_dice),
                    'fn_rate_mean': np.mean(boundary_fn),
                    'fn_rate_std': np.std(boundary_fn)
                }
        
        return stats
    
    def get_size_correlation(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute correlation between organ size and Dice score.
        
        Returns:
            Dictionary with correlation coefficient and p-value per class
        """
        from scipy import stats as scipy_stats
        
        correlations = {}
        
        for class_name in self.class_names:
            if len(self.size_vs_dice[class_name]) < 10:
                continue
            
            sizes, dices = zip(*self.size_vs_dice[class_name])
            corr, p_value = scipy_stats.pearsonr(sizes, dices)
            
            correlations[class_name] = {
                'correlation': corr,
                'p_value': p_value,
                'n_samples': len(sizes)
            }
        
        return correlations
    
    def get_failure_summary(self) -> Dict:
        """
        Get summary of failure cases.
        
        Returns:
            Dictionary with failure case analysis
        """
        if not self.failure_cases:
            return {'n_failures': 0}
        
        summary = {
            'n_failures': len(self.failure_cases),
            'failure_rate': len(self.failure_cases) / len(self.per_sample_errors),
            'per_class_failures': {}
        }
        
        for class_name in self.class_names:
            class_failures = [f for f in self.failure_cases 
                           if class_name in f and f[class_name]['dice'] < 0.5 
                           and f[class_name]['gt_pixels'] > 100]
            
            if class_failures:
                dice_scores = [f[class_name]['dice'] for f in class_failures]
                summary['per_class_failures'][class_name] = {
                    'count': len(class_failures),
                    'mean_dice': np.mean(dice_scores),
                    'worst_dice': np.min(dice_scores)
                }
        
        return summary
    
    def generate_report(self) -> str:
        """
        Generate comprehensive error analysis report.
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"\nTotal samples analyzed: {len(self.per_sample_errors)}")
        
        # Per-class statistics
        stats = self.get_class_statistics()
        lines.append("\n" + "-" * 70)
        lines.append("PER-CLASS STATISTICS")
        lines.append("-" * 70)
        
        for class_name, class_stats in stats.items():
            lines.append(f"\n{class_name}:")
            dice = class_stats['dice']
            lines.append(f"  Dice Score: {dice['mean']:.4f} ± {dice['std']:.4f}")
            lines.append(f"    Range: [{dice['min']:.4f}, {dice['max']:.4f}]")
            lines.append(f"    Median: {dice['median']:.4f}, IQR: [{dice['q25']:.4f}, {dice['q75']:.4f}]")
            
            fp = class_stats['false_positive_rate']
            fn = class_stats['false_negative_rate']
            lines.append(f"  False Positive Rate: {fp['mean']:.4f} ± {fp['std']:.4f}")
            lines.append(f"  False Negative Rate: {fn['mean']:.4f} ± {fn['std']:.4f}")
            
            if 'boundary' in class_stats:
                bd = class_stats['boundary']
                lines.append(f"  Boundary Dice: {bd['dice_mean']:.4f} ± {bd['dice_std']:.4f}")
                lines.append(f"  Boundary FN Rate: {bd['fn_rate_mean']:.4f} ± {bd['fn_rate_std']:.4f}")
        
        # Size correlation
        correlations = self.get_size_correlation()
        if correlations:
            lines.append("\n" + "-" * 70)
            lines.append("SIZE vs DICE CORRELATION")
            lines.append("-" * 70)
            for class_name, corr in correlations.items():
                sig = "***" if corr['p_value'] < 0.001 else "**" if corr['p_value'] < 0.01 else "*" if corr['p_value'] < 0.05 else ""
                lines.append(f"  {class_name}: r = {corr['correlation']:.4f} (p = {corr['p_value']:.4f}) {sig}")
        
        # Failure summary
        failure_summary = self.get_failure_summary()
        lines.append("\n" + "-" * 70)
        lines.append("FAILURE ANALYSIS (Dice < 0.5)")
        lines.append("-" * 70)
        lines.append(f"  Total failures: {failure_summary['n_failures']}")
        lines.append(f"  Failure rate: {failure_summary.get('failure_rate', 0):.2%}")
        
        if failure_summary.get('per_class_failures'):
            for class_name, cf in failure_summary['per_class_failures'].items():
                lines.append(f"  {class_name}: {cf['count']} failures, worst Dice = {cf['worst_dice']:.4f}")
        
        # Confusion matrix summary
        lines.append("\n" + "-" * 70)
        lines.append("CONFUSION MATRIX (normalized)")
        lines.append("-" * 70)
        cm = self.get_confusion_matrix(normalize=True)
        class_labels = ['BG'] + self.class_names
        
        # Header
        header = "       " + "".join([f"{l:>10}" for l in class_labels])
        lines.append(header)
        lines.append("  Pred→")
        lines.append("  True↓")
        
        for i, label in enumerate(class_labels):
            row = f"  {label:>5} " + "".join([f"{cm[i,j]:>10.4f}" for j in range(len(class_labels))])
            lines.append(row)
        
        lines.append("\n" + "=" * 70)
        
        return '\n'.join(lines)
    
    def save_results(self, output_dir: str):
        """
        Save error analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report = self.generate_report()
        with open(output_dir / 'error_analysis_report.txt', 'w') as f:
            f.write(report)
        
        # Save statistics as JSON
        stats = {
            'class_statistics': self.get_class_statistics(),
            'size_correlation': self.get_size_correlation(),
            'failure_summary': self.get_failure_summary(),
            'n_samples': len(self.per_sample_errors)
        }
        
        # Convert numpy types for JSON serialization
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
        
        stats = convert_numpy(stats)
        
        with open(output_dir / 'error_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save confusion matrix
        np.save(output_dir / 'confusion_matrix.npy', self.confusion_matrix)
        
        # Save failure cases
        if self.failure_cases:
            failure_cases = convert_numpy(self.failure_cases)
            with open(output_dir / 'failure_cases.json', 'w') as f:
                json.dump(failure_cases, f, indent=2)
        
        print(f"Error analysis results saved to {output_dir}")


# Test
if __name__ == "__main__":
    print("Testing Error Analysis Module...")
    
    np.random.seed(42)
    
    analyzer = ErrorAnalyzer()
    
    # Simulate some predictions
    for i in range(100):
        gt = np.random.randint(0, 4, size=(256, 256))
        # Add some noise to create predictions
        pred = gt.copy()
        noise_mask = np.random.random((256, 256)) < 0.1
        pred[noise_mask] = np.random.randint(0, 4, size=np.sum(noise_mask))
        
        analyzer.update(pred, gt, sample_id=f'sample_{i}')
    
    # Generate report
    print(analyzer.generate_report())
    
    # Save results
    analyzer.save_results('/tmp/error_analysis_test')
    
    print("\nError analysis tests passed!")
