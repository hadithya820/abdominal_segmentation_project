"""
Comprehensive Metrics Module for Medical Image Segmentation
Implements: Dice Score, IoU, Pixel Accuracy, Hausdorff Distance, Statistical Tests
"""

import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for multi-organ segmentation.
    
    Supports both 2D (slice-based) and 3D (volumetric) evaluation.
    """
    
    def __init__(self, num_classes: int = 4, class_names: List[str] = None, 
                 include_background: bool = False):
        """
        Args:
            num_classes: Total number of classes (including background)
            class_names: Names for each class (excluding background if include_background=False)
            include_background: Whether to include background in metrics computation
        """
        self.num_classes = num_classes
        self.include_background = include_background
        
        if class_names is None:
            if include_background:
                self.class_names = [f'Class_{i}' for i in range(num_classes)]
            else:
                self.class_names = [f'Class_{i}' for i in range(1, num_classes)]
        else:
            self.class_names = class_names
        
        self.start_class = 0 if include_background else 1
        
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array if needed."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def dice_score(self, pred: np.ndarray, target: np.ndarray, 
                   smooth: float = 1e-7) -> Dict[str, float]:
        """
        Compute Dice Score (F1 Score) per class.
        
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        
        Args:
            pred: Predicted segmentation mask (H, W) or (D, H, W)
            target: Ground truth segmentation mask
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dictionary with per-class and mean Dice scores
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        
        results = {}
        dice_scores = []
        
        for i, class_name in enumerate(self.class_names, start=self.start_class):
            pred_class = (pred == i).astype(np.float32)
            target_class = (target == i).astype(np.float32)
            
            intersection = np.sum(pred_class * target_class)
            union = np.sum(pred_class) + np.sum(target_class)
            
            if union > 0:
                dice = (2.0 * intersection + smooth) / (union + smooth)
            else:
                dice = 1.0  # Both empty - perfect match
            
            results[class_name] = dice
            dice_scores.append(dice)
        
        results['mean'] = np.mean(dice_scores)
        return results
    
    def iou_score(self, pred: np.ndarray, target: np.ndarray, 
                  smooth: float = 1e-7) -> Dict[str, float]:
        """
        Compute Intersection over Union (Jaccard Index) per class.
        
        IoU = |X ∩ Y| / |X ∪ Y|
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            smooth: Smoothing factor
            
        Returns:
            Dictionary with per-class and mean IoU scores
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        
        results = {}
        iou_scores = []
        
        for i, class_name in enumerate(self.class_names, start=self.start_class):
            pred_class = (pred == i).astype(np.float32)
            target_class = (target == i).astype(np.float32)
            
            intersection = np.sum(pred_class * target_class)
            union = np.sum(pred_class) + np.sum(target_class) - intersection
            
            if union > 0:
                iou = (intersection + smooth) / (union + smooth)
            else:
                iou = 1.0  # Both empty - perfect match
            
            results[class_name] = iou
            iou_scores.append(iou)
        
        results['mean'] = np.mean(iou_scores)
        return results
    
    def pixel_accuracy(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Compute pixel-wise accuracy metrics.
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            
        Returns:
            Dictionary with overall and per-class accuracy
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        
        results = {}
        
        # Overall accuracy
        correct = np.sum(pred == target)
        total = pred.size
        results['overall'] = correct / total
        
        # Per-class accuracy (class-wise recall)
        for i, class_name in enumerate(self.class_names, start=self.start_class):
            target_class = (target == i)
            if np.sum(target_class) > 0:
                pred_class = (pred == i)
                correct_class = np.sum(pred_class & target_class)
                results[class_name] = correct_class / np.sum(target_class)
            else:
                results[class_name] = 1.0  # No ground truth pixels
        
        return results
    
    def hausdorff_distance(self, pred: np.ndarray, target: np.ndarray, 
                           percentile: float = 95) -> Dict[str, float]:
        """
        Compute Hausdorff Distance per class.
        
        Uses the specified percentile (default 95th) for robustness to outliers.
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            percentile: Percentile for robust Hausdorff distance (default: 95)
            
        Returns:
            Dictionary with per-class and mean Hausdorff distances
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        
        results = {}
        hd_scores = []
        
        for i, class_name in enumerate(self.class_names, start=self.start_class):
            pred_class = (pred == i).astype(np.uint8)
            target_class = (target == i).astype(np.uint8)
            
            # Get surface points (boundary pixels)
            pred_surface = self._get_surface_points(pred_class)
            target_surface = self._get_surface_points(target_class)
            
            if len(pred_surface) == 0 and len(target_surface) == 0:
                hd = 0.0  # Both empty
            elif len(pred_surface) == 0 or len(target_surface) == 0:
                hd = np.inf  # One empty, one not
            else:
                hd = self._compute_percentile_hausdorff(
                    pred_surface, target_surface, percentile
                )
            
            results[class_name] = hd
            if not np.isinf(hd):
                hd_scores.append(hd)
        
        results['mean'] = np.mean(hd_scores) if hd_scores else np.inf
        return results
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface (boundary) points from binary mask."""
        if np.sum(mask) == 0:
            return np.array([])
        
        # Erode and get boundary
        if mask.ndim == 2:
            struct = ndimage.generate_binary_structure(2, 1)
        else:
            struct = ndimage.generate_binary_structure(3, 1)
        
        eroded = ndimage.binary_erosion(mask, structure=struct)
        surface = mask.astype(bool) & ~eroded
        
        return np.array(np.where(surface)).T
    
    def _compute_percentile_hausdorff(self, points1: np.ndarray, points2: np.ndarray, 
                                       percentile: float) -> float:
        """Compute percentile Hausdorff distance between two point sets."""
        from scipy.spatial import cKDTree
        
        # Build KD-trees for efficient nearest neighbor search
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Compute distances from points1 to points2
        distances1, _ = tree2.query(points1)
        # Compute distances from points2 to points1
        distances2, _ = tree1.query(points2)
        
        # Combine and compute percentile
        all_distances = np.concatenate([distances1, distances2])
        return np.percentile(all_distances, percentile)
    
    def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray,
                           hausdorff_percentile: float = 95) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics at once.
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            hausdorff_percentile: Percentile for Hausdorff distance
            
        Returns:
            Dictionary with all metrics
        """
        return {
            'dice': self.dice_score(pred, target),
            'iou': self.iou_score(pred, target),
            'pixel_accuracy': self.pixel_accuracy(pred, target),
            'hausdorff_distance': self.hausdorff_distance(pred, target, hausdorff_percentile)
        }


class StatisticalTests:
    """
    Statistical tests for comparing model performance.
    """
    
    @staticmethod
    def paired_t_test(scores1: List[float], scores2: List[float], 
                      alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Perform paired t-test to compare two models.
        
        Args:
            scores1: Scores from model 1
            scores2: Scores from model 2
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with t-statistic, p-value, and effect size (Cohen's d)
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)
        
        # Cohen's d effect size
        diff = scores1 - scores2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_diff': np.mean(diff),
            'std_diff': np.std(diff),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def wilcoxon_test(scores1: List[float], scores2: List[float],
                      alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Args:
            scores1: Scores from model 1
            scores2: Scores from model 2
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test statistic, p-value, and effect size
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        try:
            stat, p_value = stats.wilcoxon(scores1, scores2, alternative=alternative)
        except ValueError:
            # All differences are zero
            return {
                'statistic': 0,
                'p_value': 1.0,
                'effect_size': 0,
                'significant': False
            }
        
        # Effect size: r = Z / sqrt(N)
        n = len(scores1)
        z = stats.norm.ppf(1 - p_value / 2)
        effect_size = z / np.sqrt(n)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'median_diff': np.median(scores1 - scores2),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def bootstrap_confidence_interval(scores: List[float], 
                                       confidence: float = 0.95,
                                       n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for metric scores.
        
        Args:
            scores: List of scores
            confidence: Confidence level (default: 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with mean, CI lower, CI upper
        """
        scores = np.array(scores)
        n = len(scores)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_lower': lower,
            'ci_upper': upper,
            'confidence': confidence
        }


class BatchMetricsCalculator:
    """
    Efficiently compute metrics over batches of predictions.
    """
    
    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.metrics = SegmentationMetrics(num_classes, class_names)
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_dice = {name: [] for name in self.metrics.class_names}
        self.all_iou = {name: [] for name in self.metrics.class_names}
        self.all_accuracy = {name: [] for name in self.metrics.class_names}
        self.all_hausdorff = {name: [] for name in self.metrics.class_names}
        self.overall_accuracy = []
    
    def update(self, pred: np.ndarray, target: np.ndarray, 
               compute_hausdorff: bool = True):
        """
        Update metrics with new prediction-target pair.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            compute_hausdorff: Whether to compute Hausdorff distance (slow)
        """
        # Compute metrics
        dice = self.metrics.dice_score(pred, target)
        iou = self.metrics.iou_score(pred, target)
        accuracy = self.metrics.pixel_accuracy(pred, target)
        
        # Store per-class metrics
        for name in self.metrics.class_names:
            self.all_dice[name].append(dice[name])
            self.all_iou[name].append(iou[name])
            self.all_accuracy[name].append(accuracy[name])
        
        self.overall_accuracy.append(accuracy['overall'])
        
        # Hausdorff is expensive, only compute if requested
        if compute_hausdorff:
            hd = self.metrics.hausdorff_distance(pred, target)
            for name in self.metrics.class_names:
                if not np.isinf(hd[name]):
                    self.all_hausdorff[name].append(hd[name])
    
    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all accumulated metrics.
        
        Returns:
            Dictionary with summary statistics per metric and class
        """
        summary = {
            'dice': {},
            'iou': {},
            'pixel_accuracy': {},
            'hausdorff_distance': {}
        }
        
        for name in self.metrics.class_names:
            # Dice
            scores = self.all_dice[name]
            summary['dice'][name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            }
            
            # IoU
            scores = self.all_iou[name]
            summary['iou'][name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            }
            
            # Accuracy
            scores = self.all_accuracy[name]
            summary['pixel_accuracy'][name] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            
            # Hausdorff
            scores = self.all_hausdorff.get(name, [])
            if scores:
                summary['hausdorff_distance'][name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    '95th_percentile': np.percentile(scores, 95)
                }
        
        # Overall accuracy
        summary['pixel_accuracy']['overall'] = {
            'mean': np.mean(self.overall_accuracy),
            'std': np.std(self.overall_accuracy)
        }
        
        # Mean across classes
        for metric in ['dice', 'iou']:
            all_means = [summary[metric][name]['mean'] for name in self.metrics.class_names]
            summary[metric]['mean'] = {
                'mean': np.mean(all_means),
                'std': np.std(all_means)
            }
        
        return summary
    
    def get_raw_scores(self) -> Dict[str, Dict[str, List[float]]]:
        """Get raw per-sample scores for statistical testing."""
        return {
            'dice': dict(self.all_dice),
            'iou': dict(self.all_iou),
            'pixel_accuracy': dict(self.all_accuracy),
            'hausdorff_distance': dict(self.all_hausdorff)
        }


def format_metrics_table(summary: Dict, metric_name: str = 'dice') -> str:
    """
    Format metrics summary as a nicely formatted table.
    
    Args:
        summary: Summary dictionary from BatchMetricsCalculator
        metric_name: Which metric to format ('dice', 'iou', etc.)
        
    Returns:
        Formatted string table
    """
    metric_data = summary[metric_name]
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"{metric_name.upper()} SCORES")
    lines.append("=" * 60)
    lines.append(f"{'Class':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 60)
    
    for class_name, scores in metric_data.items():
        if class_name == 'mean':
            continue
        mean = scores.get('mean', 0)
        std = scores.get('std', 0)
        min_val = scores.get('min', 0)
        max_val = scores.get('max', 0)
        lines.append(f"{class_name:<20} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")
    
    if 'mean' in metric_data:
        lines.append("-" * 60)
        mean_data = metric_data['mean']
        lines.append(f"{'OVERALL':<20} {mean_data['mean']:>10.4f} {mean_data.get('std', 0):>10.4f}")
    
    lines.append("=" * 60)
    
    return '\n'.join(lines)


# Test the metrics
if __name__ == "__main__":
    # Create dummy data
    np.random.seed(42)
    pred = np.random.randint(0, 4, size=(256, 256))
    target = np.random.randint(0, 4, size=(256, 256))
    
    # Test SegmentationMetrics
    metrics = SegmentationMetrics(
        num_classes=4, 
        class_names=['Liver', 'Kidneys', 'Spleen']
    )
    
    print("Testing Dice Score:")
    dice = metrics.dice_score(pred, target)
    for k, v in dice.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nTesting IoU Score:")
    iou = metrics.iou_score(pred, target)
    for k, v in iou.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nTesting Pixel Accuracy:")
    acc = metrics.pixel_accuracy(pred, target)
    for k, v in acc.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nTesting Hausdorff Distance:")
    hd = metrics.hausdorff_distance(pred, target)
    for k, v in hd.items():
        print(f"  {k}: {v:.4f}")
    
    # Test Statistical Tests
    print("\n" + "="*50)
    print("Testing Statistical Tests:")
    scores1 = np.random.normal(0.9, 0.05, 100)
    scores2 = np.random.normal(0.88, 0.05, 100)
    
    print("\nPaired t-test:")
    t_test = StatisticalTests.paired_t_test(scores1, scores2)
    for k, v in t_test.items():
        print(f"  {k}: {v}")
    
    print("\nWilcoxon test:")
    wilcoxon = StatisticalTests.wilcoxon_test(scores1, scores2)
    for k, v in wilcoxon.items():
        print(f"  {k}: {v}")
    
    print("\nBootstrap CI:")
    ci = StatisticalTests.bootstrap_confidence_interval(scores1)
    for k, v in ci.items():
        print(f"  {k}: {v}")
    
    print("\nAll tests passed!")
