"""
3D Training Pipeline for Volumetric Medical Image Segmentation

Features:
- Multi-model support (UNet3D, VNet, SegResNet)
- Gradient checkpointing for memory efficiency
- Mixed precision training (AMP)
- Deep supervision support
- Comprehensive logging (TensorBoard, JSON)
- Learning rate scheduling with warmup
- Multi-GPU support (DataParallel)
- Automatic batch size finding

Optimized for:
- 32GB GPU memory
- 256GB system RAM
- 64 CPU cores
- Fast training with large batch accumulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DataParallel

import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import warnings
import psutil
import gc

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.unet_3d import UNet3D
from models.vnet import VNet
from models.segresnet import SegResNet, SegResNetLite
from utils.dataset_3d import (
    AbdomenCT3DDataset, 
    create_3d_dataloaders,
    get_training_transforms_3d
)
from utils.losses import CombinedLoss, DiceLoss


class DiceLoss3D(nn.Module):
    """
    3D Dice Loss with multi-class support
    """
    def __init__(self, smooth=1.0, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - class indices
        """
        pred = torch.softmax(pred, dim=1)
        n_classes = pred.shape[1]
        
        # One-hot encode target
        target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=n_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Optionally exclude background
        start_idx = 0 if self.include_background else 1
        
        dice_scores = []
        for i in range(start_idx, n_classes):
            pred_i = pred[:, i].contiguous().view(-1)
            target_i = target_one_hot[:, i].contiguous().view(-1)
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        return 1 - torch.stack(dice_scores).mean()


class CombinedLoss3D(nn.Module):
    """Combined Dice + Cross Entropy loss for 3D segmentation"""
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None, 
                 include_background=False):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss3D(include_background=include_background)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


class DeepSupervisionLoss(nn.Module):
    """Loss function with deep supervision support"""
    
    def __init__(self, base_loss, weights=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
    
    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            total_loss = 0
            for i, (output, weight) in enumerate(zip(outputs, self.weights)):
                total_loss += weight * self.base_loss(output, target)
            return total_loss
        return self.base_loss(outputs, target)


class Metrics3D:
    """Compute 3D segmentation metrics"""
    
    def __init__(self, n_classes=4, include_background=False):
        self.n_classes = n_classes
        self.include_background = include_background
        self.reset()
    
    def reset(self):
        self.dice_scores = {i: [] for i in range(self.n_classes)}
        self.ious = {i: [] for i in range(self.n_classes)}
    
    @torch.no_grad()
    def update(self, pred, target):
        """
        Update metrics with batch predictions
        
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - class indices
        """
        pred_labels = pred.argmax(dim=1)
        
        for c in range(self.n_classes):
            pred_c = (pred_labels == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            # Dice
            if union > 0:
                dice = (2 * intersection) / union
                self.dice_scores[c].append(dice.item())
            
            # IoU
            iou_union = union - intersection
            if iou_union > 0:
                iou = intersection / iou_union
                self.ious[c].append(iou.item())
    
    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics"""
        results = {}
        
        start_idx = 0 if self.include_background else 1
        
        # Per-class Dice
        dice_values = []
        for c in range(start_idx, self.n_classes):
            if self.dice_scores[c]:
                mean_dice = np.mean(self.dice_scores[c])
                results[f'dice_class_{c}'] = mean_dice
                dice_values.append(mean_dice)
        
        # Mean Dice
        if dice_values:
            results['mean_dice'] = np.mean(dice_values)
        
        # Per-class IoU
        iou_values = []
        for c in range(start_idx, self.n_classes):
            if self.ious[c]:
                mean_iou = np.mean(self.ious[c])
                results[f'iou_class_{c}'] = mean_iou
                iou_values.append(mean_iou)
        
        # Mean IoU
        if iou_values:
            results['mean_iou'] = np.mean(iou_values)
        
        return results


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Trainer3D:
    """
    3D Training pipeline with comprehensive optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        output_path = Path(config['output_dir'])
        
        # Redirect to DBFS if running on Databricks to avoid Workspace file limits
        if Path('/dbfs').exists() and not output_path.is_absolute():
            print("Redirecting output to DBFS to avoid Workspace file limits...")
            # Use a project-specific folder in FileStore
            self.output_dir = Path('/dbfs/FileStore/abdominal_segmentation_project') / output_path
        else:
            self.output_dir = output_path
            
        print(f"Output directory: {self.output_dir}")
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Print system info
        self._print_system_info()
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        # Training state
        self.start_epoch = 0
        self.best_val_metric = 0.0
        self.epochs_no_improve = 0
        self.global_step = 0
        
        # Resume from checkpoint if specified
        if config.get('resume_from'):
            self._load_checkpoint(config['resume_from'])
    
    def _print_system_info(self):
        """Print system and GPU information"""
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        
        print(f"\nCPU cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print("="*60 + "\n")
    
    def _setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        config = self.config
        
        self.train_loader, self.val_loader = create_3d_dataloaders(
            data_dir=config['data_dir'],
            train_split=config['train_split'],
            val_split=config['val_split'],
            patch_size=tuple(config.get('patch_size', [64, 128, 128])),
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 8),
            cache_size=config.get('cache_size', 10),
            samples_per_volume=config.get('samples_per_volume', 8),
            use_augmentation=config.get('use_augmentation', True)
        )
        
        print(f"Training batches per epoch: {len(self.train_loader)}")
        print(f"Validation batches per epoch: {len(self.val_loader)}")
    
    def _setup_model(self):
        """Setup model architecture"""
        print(f"Setting up model: {self.config['model_name']}...")
        
        model_name = self.config['model_name'].lower()
        n_channels = self.config.get('n_channels', 1)
        n_classes = self.config.get('n_classes', 4)
        use_checkpoint = self.config.get('use_checkpoint', True)
        deep_supervision = self.config.get('deep_supervision', False)
        
        if model_name == 'unet3d':
            self.model = UNet3D(
                n_channels=n_channels,
                n_classes=n_classes,
                base_filters=self.config.get('base_filters', 32),
                trilinear=self.config.get('trilinear', True),
                use_checkpoint=use_checkpoint,
                deep_supervision=deep_supervision
            )
        elif model_name == 'vnet':
            self.model = VNet(
                n_channels=n_channels,
                n_classes=n_classes,
                base_filters=self.config.get('base_filters', 16),
                trilinear=self.config.get('trilinear', False),
                use_checkpoint=use_checkpoint,
                deep_supervision=deep_supervision
            )
        elif model_name == 'segresnet':
            self.model = SegResNet(
                n_channels=n_channels,
                n_classes=n_classes,
                init_filters=self.config.get('base_filters', 32),
                trilinear=self.config.get('trilinear', True),
                use_checkpoint=use_checkpoint,
                use_vae=self.config.get('use_vae', False),
                deep_supervision=deep_supervision
            )
        elif model_name == 'segresnet_lite':
            self.model = SegResNetLite(
                n_channels=n_channels,
                n_classes=n_classes,
                use_checkpoint=use_checkpoint,
                deep_supervision=deep_supervision
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    def _setup_training(self):
        """Setup loss, optimizer, scheduler, and other training components"""
        config = self.config
        
        # Loss function
        base_loss = CombinedLoss3D(
            dice_weight=config.get('dice_weight', 0.5),
            ce_weight=config.get('ce_weight', 0.5),
            include_background=config.get('include_background', False)
        )
        
        if config.get('deep_supervision', False):
            self.criterion = DeepSupervisionLoss(base_loss)
        else:
            self.criterion = base_loss
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config['epochs'],
            base_lr=config['learning_rate'],
            min_lr=config.get('min_lr', 1e-7)
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Metrics
        self.metrics = Metrics3D(
            n_classes=config.get('n_classes', 4),
            include_background=config.get('include_background', False)
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        print(f"Using AMP: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")

    def _save_atomic(self, obj, path: Path):
        """Save checkpoint atomically to avoid corruption"""
        tmp_path = path.with_suffix('.tmp')
        try:
            torch.save(obj, tmp_path)
            tmp_path.replace(path)
        except Exception as e:
            print(f"WARNING: Failed to save checkpoint to {path}: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass
    
    def _save_checkpoint(self, epoch: int, val_metric: float, is_best: bool = False):
        """Save training checkpoint"""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_epoch': self.scheduler.current_epoch,
            'best_val_metric': self.best_val_metric,
            'val_metric': val_metric,
            'config': self.config,
            'history': self.history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        self._save_atomic(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            self._save_atomic(checkpoint, best_path)
            print(f"  Saved best checkpoint (Dice: {val_metric:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.config.get('save_every', 10) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            self._save_atomic(checkpoint, epoch_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.current_epoch = checkpoint.get('scheduler_epoch', 0)
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    
                    # Handle deep supervision or dict output
                    if isinstance(outputs, dict):
                        loss = self.criterion(outputs['logits'], masks)
                        if 'ds_outputs' in outputs:
                            for ds_out in outputs['ds_outputs']:
                                loss += 0.25 * self.criterion(ds_out, masks)
                    elif isinstance(outputs, (list, tuple)):
                        loss = self.criterion(outputs, masks)
                    else:
                        loss = self.criterion(outputs, masks)
                    
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    loss = self.criterion(outputs['logits'], masks)
                elif isinstance(outputs, (list, tuple)):
                    loss = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'lr': f'{self.scheduler.get_lr():.2e}'
            })
            
            # Log to tensorboard
            self.global_step += 1
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item() * self.accumulation_steps, self.global_step)
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]")
        
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs['logits']
                    elif isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, masks) if not isinstance(self.criterion, DeepSupervisionLoss) else self.criterion.base_loss(outputs, masks)
            else:
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                elif isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                loss = self.criterion(outputs, masks) if not isinstance(self.criterion, DeepSupervisionLoss) else self.criterion.base_loss(outputs, masks)
            
            total_loss += loss.item()
            
            # Update metrics
            self.metrics.update(outputs, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Model: {self.config['model_name']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']} x {self.accumulation_steps} (accumulation)")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            val_dice = val_metrics.get('mean_dice', 0.0)
            
            # Update learning rate
            current_lr = self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/MeanDice', val_dice, epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(current_lr)
            
            # Save checkpoint
            is_best = val_dice > self.best_val_metric
            if is_best:
                self.best_val_metric = val_dice
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            self._save_checkpoint(epoch, val_dice, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            for key, value in val_metrics.items():
                if 'dice_class' in key:
                    print(f"  {key}: {value:.4f}")
            
            # Early stopping
            if self.epochs_no_improve >= self.config.get('early_stopping_patience', 20):
                print(f"\nEarly stopping triggered after {self.epochs_no_improve} epochs without improvement")
                break
            
            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation Dice: {self.best_val_metric:.4f}")
        print("="*60)
        
        # Save final history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        
        return self.best_val_metric


def profile_memory(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Profile memory usage for a given model configuration"""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    n_channels = config.get('n_channels', 1)
    n_classes = config.get('n_classes', 4)
    base_filters = config.get('base_filters', 32)
    
    if model_name.lower() == 'unet3d':
        model = UNet3D(n_channels, n_classes, base_filters, use_checkpoint=True)
    elif model_name.lower() == 'vnet':
        model = VNet(n_channels, n_classes, base_filters, use_checkpoint=True)
    elif model_name.lower() == 'segresnet':
        model = SegResNet(n_channels, n_classes, base_filters, use_checkpoint=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.train()
    
    # Test input
    batch_size = config.get('batch_size', 2)
    patch_size = config.get('patch_size', [64, 128, 128])
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    x = torch.randn(batch_size, n_channels, *patch_size).to(device)
    target = torch.randint(0, n_classes, (batch_size, *patch_size)).to(device)
    
    # Forward pass
    start_time = time.time()
    with autocast('cuda'):
        y = model(x)
        if isinstance(y, dict):
            y = y['logits']
        loss = nn.CrossEntropyLoss()(y, target)
    forward_time = time.time() - start_time
    
    # Backward pass
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    # Model size
    model_params = sum(p.numel() for p in model.parameters())
    model_size = model_params * 4 / 1024**2  # MB
    
    del model, x, target, y, loss
    torch.cuda.empty_cache()
    
    return {
        'model': model_name,
        'batch_size': batch_size,
        'patch_size': patch_size,
        'parameters': model_params,
        'model_size_mb': model_size,
        'peak_memory_gb': peak_memory,
        'forward_time_s': forward_time,
        'backward_time_s': backward_time,
        'total_time_s': forward_time + backward_time
    }


def main():
    parser = argparse.ArgumentParser(description='Train 3D segmentation models')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--profile', action='store_true', help='Profile memory usage only')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    if args.profile:
        # Just profile memory
        print("\n" + "="*60)
        print("MEMORY PROFILING")
        print("="*60)
        
        profile_results = profile_memory(config['model_name'], config)
        
        print(f"\nModel: {profile_results['model']}")
        print(f"Batch size: {profile_results['batch_size']}")
        print(f"Patch size: {profile_results['patch_size']}")
        print(f"Parameters: {profile_results['parameters']:,}")
        print(f"Model size: {profile_results['model_size_mb']:.1f} MB")
        print(f"Peak GPU memory: {profile_results['peak_memory_gb']:.2f} GB")
        print(f"Forward time: {profile_results['forward_time_s']*1000:.1f} ms")
        print(f"Backward time: {profile_results['backward_time_s']*1000:.1f} ms")
        
        # Save profile results
        profile_path = Path(config.get('output_dir', 'results')) / 'memory_profile.json'
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, 'w') as f:
            json.dump(profile_results, f, indent=2)
        print(f"\nProfile saved to {profile_path}")
    else:
        # Full training
        trainer = Trainer3D(config)
        trainer.train()


if __name__ == "__main__":
    main()
