import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from models.unet_2d import UNet2D
from utils.dataset import AbdomenCTDataset, get_training_augmentation, get_validation_augmentation
from utils.losses import CombinedLoss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Create datasets
        print("Loading datasets...")
        self.train_dataset = AbdomenCTDataset(
            data_dir=config['data_dir'],
            split_file=config['train_split'],
            transform=get_training_augmentation(),
            cache_data=config.get('cache_data', False)
        )
        
        self.val_dataset = AbdomenCTDataset(
            data_dir=config['data_dir'],
            split_file=config['val_split'],
            transform=get_validation_augmentation(),
            cache_data=config.get('cache_data', False)
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        print(f"Train dataset: {len(self.train_dataset)} slices")
        print(f"Val dataset: {len(self.val_dataset)} slices")
        
        # Create model
        print("Creating model...")
        self.model = UNet2D(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config.get('bilinear', False)
        ).to(self.device)
        
        # Loss function
        self.criterion = CombinedLoss(
            dice_weight=config.get('dice_weight', 0.5),
            ce_weight=config.get('ce_weight', 0.5)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP) training")
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_train_loss = train_loss / len(self.train_loader)
        return avg_train_loss
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                val_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss: {val_loss:.4f}")
    
    def train(self):
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.epochs_no_improve >= self.config.get('early_stopping_patience', 15):
                print(f"\nEarly stopping triggered after {self.epochs_no_improve} epochs without improvement")
                break
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*50)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for abdominal organ segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Print config
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()