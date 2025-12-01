"""
Enhanced Visualization Suite for Multi-Organ Segmentation

Features:
- 2D overlay predictions (CT + ground truth + prediction)
- Model comparison grids (side-by-side)
- Error analysis visualizations
- Metrics distribution plots
- 3D volume renderings (using Plotly)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))


class SegmentationVisualizer:
    """
    Comprehensive visualization tools for segmentation results.
    """
    
    def __init__(self, class_names: List[str] = None, 
                 colors: List[Tuple] = None,
                 figsize_base: Tuple[int, int] = (4, 4)):
        """
        Initialize visualizer with class definitions.
        
        Args:
            class_names: Names for each organ class
            colors: RGB colors for each class (0-255)
            figsize_base: Base figure size per subplot
        """
        self.class_names = class_names or ['Background', 'Liver', 'Kidneys', 'Spleen']
        self.colors = colors or [
            (0, 0, 0),       # Background - Black
            (255, 0, 0),     # Liver - Red
            (0, 255, 0),     # Kidneys - Green
            (0, 0, 255),     # Spleen - Blue
        ]
        self.figsize_base = figsize_base
        
        # Create colormap
        self.cmap = ListedColormap(np.array(self.colors) / 255.0)
    
    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to RGB image."""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.colors):
            rgb[mask == class_idx] = color
        return rgb
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """Create overlay of mask on grayscale image."""
        # Normalize image to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to RGB
        image_rgb = np.stack([image, image, image], axis=-1)
        
        # Create mask RGB
        mask_rgb = self.mask_to_rgb(mask)
        
        # Create overlay (only where mask is not background)
        overlay = image_rgb.copy().astype(np.float32)
        non_bg = mask > 0
        overlay[non_bg] = (1 - alpha) * image_rgb[non_bg] + alpha * mask_rgb[non_bg]
        
        return overlay.astype(np.uint8)
    
    def visualize_single_prediction(self, image: np.ndarray, 
                                    ground_truth: np.ndarray,
                                    prediction: np.ndarray,
                                    title: str = None,
                                    save_path: str = None,
                                    show: bool = True) -> plt.Figure:
        """
        Visualize a single prediction with CT, GT, prediction, and overlay.
        
        Args:
            image: CT slice (H, W)
            ground_truth: Ground truth mask (H, W)
            prediction: Predicted mask (H, W)
            title: Optional title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # CT Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('CT Input', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Ground Truth
        gt_rgb = self.mask_to_rgb(ground_truth)
        axes[1].imshow(gt_rgb)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        pred_rgb = self.mask_to_rgb(prediction)
        axes[2].imshow(pred_rgb)
        axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Overlay
        overlay = self.create_overlay(image, prediction)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        # Legend
        legend_elements = [mpatches.Patch(facecolor=np.array(self.colors[i])/255, 
                                          label=self.class_names[i])
                          for i in range(1, len(self.class_names))]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=10, frameon=True)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_model_comparison(self, image: np.ndarray,
                                   ground_truth: np.ndarray,
                                   predictions: Dict[str, np.ndarray],
                                   save_path: str = None,
                                   show: bool = True) -> plt.Figure:
        """
        Compare predictions from multiple models side-by-side.
        
        Args:
            image: CT slice
            ground_truth: Ground truth mask
            predictions: Dictionary mapping model names to predictions
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(2, n_models + 1, 
                                figsize=((n_models + 1) * 4, 8))
        
        # First row: CT and predictions
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('CT Input', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        for idx, (model_name, pred) in enumerate(predictions.items(), 1):
            pred_rgb = self.mask_to_rgb(pred)
            axes[0, idx].imshow(pred_rgb)
            axes[0, idx].set_title(model_name, fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
        
        # Second row: Ground truth and overlays
        gt_rgb = self.mask_to_rgb(ground_truth)
        axes[1, 0].imshow(gt_rgb)
        axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        for idx, (model_name, pred) in enumerate(predictions.items(), 1):
            overlay = self.create_overlay(image, pred)
            axes[1, idx].imshow(overlay)
            axes[1, idx].set_title(f'{model_name} Overlay', fontsize=10)
            axes[1, idx].axis('off')
        
        # Legend
        legend_elements = [mpatches.Patch(facecolor=np.array(self.colors[i])/255, 
                                          label=self.class_names[i])
                          for i in range(1, len(self.class_names))]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=10, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_error_map(self, ground_truth: np.ndarray,
                           prediction: np.ndarray,
                           image: np.ndarray = None,
                           save_path: str = None,
                           show: bool = True) -> plt.Figure:
        """
        Visualize segmentation errors.
        
        Shows:
        - Correct predictions (green)
        - False positives (red)
        - False negatives (blue)
        
        Args:
            ground_truth: Ground truth mask
            prediction: Predicted mask
            image: Optional CT image for background
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Create error map
        error_map = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
        
        # For each organ class
        for class_idx in range(1, len(self.class_names)):
            gt_class = ground_truth == class_idx
            pred_class = prediction == class_idx
            
            # True positives (correct) - green
            tp = gt_class & pred_class
            error_map[tp] = [0, 255, 0]
            
            # False positives (predicted but not in GT) - red
            fp = pred_class & ~gt_class
            error_map[fp] = [255, 0, 0]
            
            # False negatives (in GT but not predicted) - blue
            fn = gt_class & ~pred_class
            error_map[fn] = [0, 0, 255]
        
        # CT Image
        if image is not None:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(np.zeros_like(ground_truth), cmap='gray')
        axes[0].set_title('CT Input', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(self.mask_to_rgb(ground_truth))
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(self.mask_to_rgb(prediction))
        axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Error Map
        axes[3].imshow(error_map)
        axes[3].set_title('Error Map', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        # Error legend
        error_legend = [
            mpatches.Patch(facecolor='green', label='Correct (TP)'),
            mpatches.Patch(facecolor='red', label='False Positive'),
            mpatches.Patch(facecolor='blue', label='False Negative')
        ]
        fig.legend(handles=error_legend, loc='lower center', 
                  ncol=3, fontsize=10, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_grid(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      n_cols: int = 4,
                      save_path: str = None,
                      show: bool = True) -> plt.Figure:
        """
        Visualize multiple samples in a grid format.
        
        Args:
            samples: List of (image, ground_truth, prediction) tuples
            n_cols: Number of columns in grid
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        n_samples = len(samples)
        n_rows = n_samples
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        
        column_titles = ['CT Input', 'Ground Truth', 'Prediction', 'Overlay']
        
        for row_idx, (image, gt, pred) in enumerate(samples):
            # CT
            axes[row_idx, 0].imshow(image, cmap='gray')
            if row_idx == 0:
                axes[row_idx, 0].set_title(column_titles[0], fontsize=12, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            # Ground Truth
            axes[row_idx, 1].imshow(self.mask_to_rgb(gt))
            if row_idx == 0:
                axes[row_idx, 1].set_title(column_titles[1], fontsize=12, fontweight='bold')
            axes[row_idx, 1].axis('off')
            
            # Prediction
            axes[row_idx, 2].imshow(self.mask_to_rgb(pred))
            if row_idx == 0:
                axes[row_idx, 2].set_title(column_titles[2], fontsize=12, fontweight='bold')
            axes[row_idx, 2].axis('off')
            
            # Overlay
            overlay = self.create_overlay(image, pred)
            axes[row_idx, 3].imshow(overlay)
            if row_idx == 0:
                axes[row_idx, 3].set_title(column_titles[3], fontsize=12, fontweight='bold')
            axes[row_idx, 3].axis('off')
        
        # Legend
        legend_elements = [mpatches.Patch(facecolor=np.array(self.colors[i])/255, 
                                          label=self.class_names[i])
                          for i in range(1, len(self.class_names))]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=10, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.04)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


class MetricsVisualizer:
    """
    Visualizations for metrics and statistical comparisons.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """Initialize with plotting style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-whitegrid')
    
    def plot_dice_distribution(self, scores_dict: Dict[str, Dict[str, List[float]]],
                               save_path: str = None,
                               show: bool = True) -> plt.Figure:
        """
        Plot distribution of Dice scores per model and class.
        
        Args:
            scores_dict: Dictionary of model -> class -> scores
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        model_names = list(scores_dict.keys())
        n_models = len(model_names)
        
        # Get class names from first model
        class_names = list(scores_dict[model_names[0]].keys())
        n_classes = len(class_names)
        
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
        if n_classes == 1:
            axes = [axes]
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        
        for idx, class_name in enumerate(class_names):
            ax = axes[idx]
            
            data = []
            labels = []
            for model_name in model_names:
                scores = scores_dict[model_name].get(class_name, [])
                data.append(scores)
                labels.append(model_name)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{class_name} Dice Distribution', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dice Score')
            ax.set_ylim([0, 1.05])
            ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 threshold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_metrics_comparison(self, summary_dict: Dict[str, Dict],
                                metric: str = 'dice',
                                save_path: str = None,
                                show: bool = True) -> plt.Figure:
        """
        Bar chart comparing metrics across models.
        
        Args:
            summary_dict: Dictionary of model -> summary statistics
            metric: Metric to plot ('dice', 'iou', etc.)
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        model_names = list(summary_dict.keys())
        
        # Get class names from first model
        first_summary = summary_dict[model_names[0]][metric]
        class_names = [k for k in first_summary.keys() if k != 'mean']
        
        x = np.arange(len(class_names))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for idx, model_name in enumerate(model_names):
            means = []
            stds = []
            for class_name in class_names:
                scores = summary_dict[model_name][metric].get(class_name, {})
                means.append(scores.get('mean', 0))
                stds.append(scores.get('std', 0))
            
            offset = (idx - len(model_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, 
                         label=model_name, color=colors[idx],
                         capsize=3, error_kw={'linewidth': 1})
        
        ax.set_xlabel('Organ', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_title(f'{metric.upper()} Comparison Across Models', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_training_curves(self, history_dict: Dict[str, Dict],
                            save_path: str = None,
                            show: bool = True) -> plt.Figure:
        """
        Plot training curves for multiple models.
        
        Args:
            history_dict: Dictionary of model -> training history
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(history_dict)))
        
        for idx, (model_name, history) in enumerate(history_dict.items()):
            color = colors[idx]
            
            # Training loss
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label=f'{model_name} (train)',
                           color=color, linestyle='-', linewidth=2)
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label=f'{model_name} (val)',
                           color=color, linestyle='--', linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        for idx, (model_name, history) in enumerate(history_dict.items()):
            if 'learning_rate' in history:
                axes[1].plot(history['learning_rate'], label=model_name,
                           color=colors[idx], linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


def create_3d_volume_rendering(volume: np.ndarray, 
                               segmentation: np.ndarray,
                               save_path: str = None) -> None:
    """
    Create 3D volume rendering using Plotly.
    
    Args:
        volume: 3D CT volume (D, H, W)
        segmentation: 3D segmentation mask (D, H, W)
        save_path: Path to save HTML file
    """
    try:
        import plotly.graph_objects as go
        from skimage import measure
    except ImportError:
        print("Plotly and scikit-image required for 3D rendering.")
        print("Install with: pip install plotly scikit-image")
        return
    
    fig = go.Figure()
    
    organ_colors = {
        1: 'red',      # Liver
        2: 'green',    # Kidneys
        3: 'blue'      # Spleen
    }
    organ_names = {
        1: 'Liver',
        2: 'Kidneys',
        3: 'Spleen'
    }
    
    # Add each organ as a mesh
    for organ_id, color in organ_colors.items():
        organ_mask = segmentation == organ_id
        
        if organ_mask.sum() < 100:  # Skip if too few voxels
            continue
        
        try:
            # Get surface mesh using marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                organ_mask.astype(float), 
                level=0.5,
                spacing=(1, 1, 1)
            )
            
            # Create mesh
            x, y, z = verts.T
            i, j, k = faces.T
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=color,
                opacity=0.6,
                name=organ_names[organ_id]
            ))
        except Exception as e:
            print(f"Could not render {organ_names[organ_id]}: {e}")
    
    fig.update_layout(
        title='3D Organ Segmentation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved 3D rendering to {save_path}")
    else:
        fig.show()


# Test visualizations
if __name__ == "__main__":
    print("Testing Visualization Module...")
    
    # Create dummy data
    np.random.seed(42)
    image = np.random.rand(256, 256)
    ground_truth = np.random.randint(0, 4, size=(256, 256))
    prediction = np.random.randint(0, 4, size=(256, 256))
    
    # Test SegmentationVisualizer
    viz = SegmentationVisualizer()
    
    print("\nTesting single prediction visualization...")
    viz.visualize_single_prediction(
        image, ground_truth, prediction,
        title="Test Visualization",
        show=False,
        save_path="/tmp/test_single.png"
    )
    
    print("\nTesting error map...")
    viz.visualize_error_map(
        ground_truth, prediction, image,
        show=False,
        save_path="/tmp/test_error.png"
    )
    
    print("\nTesting model comparison...")
    predictions = {
        'UNet': prediction,
        'Attention UNet': np.random.randint(0, 4, size=(256, 256))
    }
    viz.visualize_model_comparison(
        image, ground_truth, predictions,
        show=False,
        save_path="/tmp/test_comparison.png"
    )
    
    # Test MetricsVisualizer
    metrics_viz = MetricsVisualizer()
    
    print("\nTesting Dice distribution plot...")
    scores_dict = {
        'UNet': {
            'Liver': list(np.random.normal(0.94, 0.05, 100)),
            'Kidneys': list(np.random.normal(0.93, 0.06, 100)),
            'Spleen': list(np.random.normal(0.98, 0.02, 100))
        },
        'Attention UNet': {
            'Liver': list(np.random.normal(0.95, 0.04, 100)),
            'Kidneys': list(np.random.normal(0.94, 0.05, 100)),
            'Spleen': list(np.random.normal(0.99, 0.01, 100))
        }
    }
    metrics_viz.plot_dice_distribution(
        scores_dict,
        show=False,
        save_path="/tmp/test_distribution.png"
    )
    
    print("\nAll visualization tests passed!")
