import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np

class TrainingHistoryPlotter:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.data = self.load_tensorboard_logs()
    
    def load_tensorboard_logs(self):
        """Load data from TensorBoard logs"""
        print(f"Loading logs from {self.log_dir}...")
        
        # Find event files
        event_files = list(self.log_dir.glob('events.out.tfevents.*'))
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files found in {self.log_dir}")
        
        # Load the most recent event file
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading: {event_file.name}")
        
        # Create event accumulator
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        # Extract data
        data = {}
        
        # Get available tags
        scalar_tags = ea.Tags()['scalars']
        print(f"Found tags: {scalar_tags}")
        
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': steps, 'values': values}
        
        return data
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Batch-level training loss
        if 'Train/BatchLoss' in self.data:
            batch_data = self.data['Train/BatchLoss']
            axes[0].plot(batch_data['steps'], batch_data['values'], 
                        alpha=0.3, color='blue', linewidth=0.5)
            
            # Add smoothed version
            window = 100
            smoothed = pd.Series(batch_data['values']).rolling(window=window, center=True).mean()
            axes[0].plot(batch_data['steps'], smoothed, 
                        color='blue', linewidth=2, label='Smoothed')
            
            axes[0].set_xlabel('Training Steps', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training Loss (Batch-level)', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Epoch-level losses
        if 'Train/EpochLoss' in self.data and 'Val/Loss' in self.data:
            train_data = self.data['Train/EpochLoss']
            val_data = self.data['Val/Loss']
            
            axes[1].plot(train_data['steps'], train_data['values'], 
                        marker='o', linewidth=2, label='Train Loss', color='blue')
            axes[1].plot(val_data['steps'], val_data['values'], 
                        marker='s', linewidth=2, label='Val Loss', color='orange')
            
            # Mark best validation loss
            best_idx = np.argmin(val_data['values'])
            best_epoch = val_data['steps'][best_idx]
            best_val_loss = val_data['values'][best_idx]
            
            axes[1].axvline(x=best_epoch, color='red', linestyle='--', 
                          alpha=0.7, label=f'Best (Epoch {best_epoch})')
            axes[1].plot(best_epoch, best_val_loss, 'r*', markersize=15)
            
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Training & Validation Loss (Epoch-level)', 
                            fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Add text with best value
            axes[1].text(0.05, 0.95, f'Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}',
                        transform=axes[1].transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved loss plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_learning_rate(self, save_path=None):
        """Plot learning rate schedule"""
        if 'Train/LearningRate' not in self.data:
            print("No learning rate data found")
            return
        
        lr_data = self.data['Train/LearningRate']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(lr_data['steps'], lr_data['values'], 
               linewidth=2, color='green', marker='o')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning rate plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_summary(self, save_path=None):
        """Create comprehensive summary plot"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Training loss (batch-level) with smoothing
        ax1 = fig.add_subplot(gs[0, :])
        if 'Train/BatchLoss' in self.data:
            batch_data = self.data['Train/BatchLoss']
            ax1.plot(batch_data['steps'], batch_data['values'], 
                    alpha=0.2, color='blue', linewidth=0.5, label='Raw')
            
            window = 100
            smoothed = pd.Series(batch_data['values']).rolling(window=window, center=True).mean()
            ax1.plot(batch_data['steps'], smoothed, 
                    color='blue', linewidth=2, label='Smoothed')
            
            ax1.set_xlabel('Training Steps', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Training Loss (Batch-level)', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Epoch-level losses
        ax2 = fig.add_subplot(gs[1, 0])
        if 'Train/EpochLoss' in self.data and 'Val/Loss' in self.data:
            train_data = self.data['Train/EpochLoss']
            val_data = self.data['Val/Loss']
            
            ax2.plot(train_data['steps'], train_data['values'], 
                    marker='o', linewidth=2, label='Train', color='blue', markersize=4)
            ax2.plot(val_data['steps'], val_data['values'], 
                    marker='s', linewidth=2, label='Validation', color='orange', markersize=4)
            
            best_idx = np.argmin(val_data['values'])
            best_epoch = val_data['steps'][best_idx]
            best_val_loss = val_data['values'][best_idx]
            ax2.plot(best_epoch, best_val_loss, 'r*', markersize=15, label='Best')
            
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Loss', fontsize=11)
            ax2.set_title('Train vs Validation Loss', fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate
        ax3 = fig.add_subplot(gs[1, 1])
        if 'Train/LearningRate' in self.data:
            lr_data = self.data['Train/LearningRate']
            ax3.plot(lr_data['steps'], lr_data['values'], 
                    linewidth=2, color='green', marker='o', markersize=4)
            ax3.set_xlabel('Epoch', fontsize=11)
            ax3.set_ylabel('Learning Rate', fontsize=11)
            ax3.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 4. Loss distribution
        ax4 = fig.add_subplot(gs[2, 0])
        if 'Train/EpochLoss' in self.data and 'Val/Loss' in self.data:
            train_losses = self.data['Train/EpochLoss']['values']
            val_losses = self.data['Val/Loss']['values']
            
            ax4.hist(train_losses, bins=20, alpha=0.6, label='Train', color='blue')
            ax4.hist(val_losses, bins=20, alpha=0.6, label='Validation', color='orange')
            ax4.set_xlabel('Loss Value', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Loss Distribution', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Training statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        if 'Train/EpochLoss' in self.data and 'Val/Loss' in self.data:
            train_data = self.data['Train/EpochLoss']
            val_data = self.data['Val/Loss']
            
            stats_text = f"""
            Training Statistics:
            
            Total Epochs: {len(train_data['values'])}
            
            Training Loss:
              Initial: {train_data['values'][0]:.4f}
              Final: {train_data['values'][-1]:.4f}
              Min: {min(train_data['values']):.4f}
              Mean: {np.mean(train_data['values']):.4f}
            
            Validation Loss:
              Initial: {val_data['values'][0]:.4f}
              Final: {val_data['values'][-1]:.4f}
              Best: {min(val_data['values']):.4f}
              (Epoch {np.argmin(val_data['values'])})
              Mean: {np.mean(val_data['values']):.4f}
            
            Improvement:
              Train: {((train_data['values'][0] - train_data['values'][-1]) / train_data['values'][0] * 100):.1f}%
              Val: {((val_data['values'][0] - val_data['values'][-1]) / val_data['values'][0] * 100):.1f}%
            """
            
            ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle('Training Summary', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved summary plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_training_stats(self, save_path):
        """Save training statistics to JSON"""
        stats = {}
        
        if 'Train/EpochLoss' in self.data:
            train_data = self.data['Train/EpochLoss']
            stats['train'] = {
                'epochs': len(train_data['values']),
                'initial_loss': float(train_data['values'][0]),
                'final_loss': float(train_data['values'][-1]),
                'min_loss': float(min(train_data['values'])),
                'mean_loss': float(np.mean(train_data['values'])),
                'std_loss': float(np.std(train_data['values']))
            }
        
        if 'Val/Loss' in self.data:
            val_data = self.data['Val/Loss']
            best_idx = int(np.argmin(val_data['values']))
            stats['validation'] = {
                'epochs': len(val_data['values']),
                'initial_loss': float(val_data['values'][0]),
                'final_loss': float(val_data['values'][-1]),
                'best_loss': float(val_data['values'][best_idx]),
                'best_epoch': int(val_data['steps'][best_idx]),
                'mean_loss': float(np.mean(val_data['values'])),
                'std_loss': float(np.std(val_data['values']))
            }
        
        with open(save_path, 'w') as f:
            json.dump(stats, indent=2, fp=f)
        
        print(f"Saved training statistics to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training history from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Path to TensorBoard log directory')
    parser.add_argument('--output_dir', type=str, default='results/training_plots',
                       help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['losses', 'lr', 'summary', 'all'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plotter
    plotter = TrainingHistoryPlotter(args.log_dir)
    
    # Generate plots
    if args.plot_type in ['losses', 'all']:
        plotter.plot_losses(save_path=output_dir / 'losses.png')
    
    if args.plot_type in ['lr', 'all']:
        plotter.plot_learning_rate(save_path=output_dir / 'learning_rate.png')
    
    if args.plot_type in ['summary', 'all']:
        plotter.plot_summary(save_path=output_dir / 'training_summary.png')
    
    # Save statistics
    plotter.save_training_stats(save_path=output_dir / 'training_stats.json')
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
