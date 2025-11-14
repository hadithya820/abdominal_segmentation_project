import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import random
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.unet_2d import UNet2D
from utils.dataset import AbdomenCTDataset, get_validation_augmentation

class ResultVisualizer:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("Loading model...")
        self.model = UNet2D(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config.get('bilinear', False)
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        
        # Load validation dataset
        print("Loading validation dataset...")
        self.val_dataset = AbdomenCTDataset(
            data_dir=config['data_dir'],
            split_file=config['val_split'],
            transform=get_validation_augmentation(),
            cache_data=False
        )
        
        # Class names and colors
        self.class_names = ['Background', 'Liver', 'Right Kidney', 'Left Kidney', 'Spleen']
        self.colors = [
            [0, 0, 0],       # Background - Black
            [255, 0, 0],     # Liver - Red
            [0, 255, 0],     # Right Kidney - Green
            [0, 0, 255],     # Left Kidney - Blue
            [255, 255, 0]    # Spleen - Yellow
        ]
    
    def mask_to_rgb(self, mask):
        """Convert class mask to RGB image"""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx in range(len(self.colors)):
            rgb[mask == class_idx] = self.colors[class_idx]
        return rgb
    
    def predict(self, image):
        """Generate prediction for a single image"""
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        return pred
    
    def visualize_sample(self, idx, save_path=None):
        """Visualize a single sample with prediction"""
        # Get sample
        image, mask = self.val_dataset[idx]
        
        # Generate prediction
        pred = self.predict(image)
        
        # Convert to numpy
        image_np = image.squeeze().cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Convert masks to RGB
        mask_rgb = self.mask_to_rgb(mask_np)
        pred_rgb = self.mask_to_rgb(pred)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Input CT Slice', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(mask_rgb)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_rgb)
        axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(image_np, cmap='gray')
        axes[3].imshow(pred_rgb, alpha=0.5)
        axes[3].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=np.array(self.colors[i])/255, 
                                label=self.class_names[i]) 
                          for i in range(1, len(self.class_names))]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=12, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_grid(self, num_samples=8, save_path=None):
        """Visualize multiple samples in a grid"""
        # Randomly select samples
        indices = random.sample(range(len(self.val_dataset)), num_samples)
        
        rows = num_samples
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        
        for i, idx in enumerate(indices):
            # Get sample
            image, mask = self.val_dataset[idx]
            pred = self.predict(image)
            
            # Convert to numpy
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.cpu().numpy()
            mask_rgb = self.mask_to_rgb(mask_np)
            pred_rgb = self.mask_to_rgb(pred)
            
            # Plot
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('Input' if i == 0 else '', fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_rgb)
            axes[i, 1].set_title('Ground Truth' if i == 0 else '', fontsize=12)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title('Prediction' if i == 0 else '', fontsize=12)
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(image_np, cmap='gray')
            axes[i, 3].imshow(pred_rgb, alpha=0.5)
            axes[i, 3].set_title('Overlay' if i == 0 else '', fontsize=12)
            axes[i, 3].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=np.array(self.colors[i])/255, 
                                label=self.class_names[i]) 
                          for i in range(1, len(self.class_names))]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=12, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved grid visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_case(self, case_id, max_slices=16, save_path=None):
        """Visualize multiple slices from a single case"""
        # Find slices from this case
        case_slices = [i for i, (cid, _) in enumerate(self.val_dataset.slices) 
                      if cid == case_id]
        
        if not case_slices:
            print(f"Case {case_id} not found in validation set")
            return
        
        # Sample slices evenly
        if len(case_slices) > max_slices:
            step = len(case_slices) // max_slices
            case_slices = case_slices[::step][:max_slices]
        
        num_slices = len(case_slices)
        cols = 4
        rows = num_slices
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(case_slices):
            image, mask = self.val_dataset[idx]
            pred = self.predict(image)
            
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.cpu().numpy()
            mask_rgb = self.mask_to_rgb(mask_np)
            pred_rgb = self.mask_to_rgb(pred)
            
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title(f'Slice {i+1}' if i == 0 else '', fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_rgb)
            axes[i, 1].set_title('Ground Truth' if i == 0 else '', fontsize=12)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title('Prediction' if i == 0 else '', fontsize=12)
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(image_np, cmap='gray')
            axes[i, 3].imshow(pred_rgb, alpha=0.5)
            axes[i, 3].set_title('Overlay' if i == 0 else '', fontsize=12)
            axes[i, 3].axis('off')
        
        fig.suptitle(f'Case: {case_id}', fontsize=16, fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=np.array(self.colors[i])/255, 
                                label=self.class_names[i]) 
                          for i in range(1, len(self.class_names))]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=len(legend_elements), fontsize=12, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.98, bottom=0.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved case visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize U-Net segmentation results')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/visualizations', 
                       help='Directory to save visualizations')
    parser.add_argument('--mode', type=str, default='grid', 
                       choices=['single', 'grid', 'case'],
                       help='Visualization mode')
    parser.add_argument('--num_samples', type=int, default=8, 
                       help='Number of samples for grid mode')
    parser.add_argument('--sample_idx', type=int, default=0, 
                       help='Sample index for single mode')
    parser.add_argument('--case_id', type=str, default=None, 
                       help='Case ID for case mode')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizer
    visualizer = ResultVisualizer(config, args.checkpoint)
    
    # Generate visualizations
    if args.mode == 'single':
        save_path = output_dir / f'sample_{args.sample_idx}.png'
        visualizer.visualize_sample(args.sample_idx, save_path=save_path)
    
    elif args.mode == 'grid':
        save_path = output_dir / f'grid_{args.num_samples}_samples.png'
        visualizer.visualize_grid(num_samples=args.num_samples, save_path=save_path)
    
    elif args.mode == 'case':
        if args.case_id is None:
            # Get first case from validation set
            case_id = visualizer.val_dataset.slices[0][0]
            print(f"No case_id provided, using first case: {case_id}")
        else:
            case_id = args.case_id
        
        save_path = output_dir / f'case_{case_id}.png'
        visualizer.visualize_case(case_id, save_path=save_path)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
