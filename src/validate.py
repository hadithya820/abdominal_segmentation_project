import torch
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models.unet_2d import UNet2D
from utils.dataset import AbdomenCTDataset, get_validation_augmentation

def validate_model(config_path, checkpoint_path):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = UNet2D(
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        bilinear=config.get('bilinear', False)
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = AbdomenCTDataset(
        data_dir=config['data_dir'],
        split_file=config['val_split'],
        transform=get_validation_augmentation(),
        cache_data=config.get('cache_data', True)  # Use cache if available
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True  # Enable faster data transfer
    )
    
    # Validate
    print("Running validation...")
    organ_names = ['Liver', 'Kidneys', 'Spleen']
    all_dice_scores = {organ: [] for organ in organ_names}
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device, non_blocking=True)  # Use non_blocking
            masks = masks.to(device, non_blocking=True)   # Use non_blocking
            
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)  # (B, H, W)
            
            # Vectorized dice calculation per sample per class
            batch_size = pred.shape[0]
            
            for cls_idx, organ_name in enumerate(organ_names, start=1):
                pred_cls = (pred == cls_idx).float()  # (B, H, W)
                target_cls = (masks == cls_idx).float()  # (B, H, W)
                
                # Calculate dice for all samples at once
                intersection = (pred_cls * target_cls).sum(dim=[1, 2])  # (B,)
                union = pred_cls.sum(dim=[1, 2]) + target_cls.sum(dim=[1, 2])  # (B,)
                
                # Handle empty cases
                dice = torch.where(
                    union > 0,
                    (2. * intersection) / (union + 1e-8),
                    torch.ones_like(union)
                )
                
                all_dice_scores[organ_name].extend(dice.cpu().numpy().tolist())
    
    # Print results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    mean_dice_all = []
    for organ_name in organ_names:
        scores = all_dice_scores[organ_name]
        mean_dice = np.mean(scores)
        std_dice = np.std(scores)
        mean_dice_all.append(mean_dice)
        
        print(f"{organ_name}:")
        print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"  Min Dice: {np.min(scores):.4f}")
        print(f"  Max Dice: {np.max(scores):.4f}")
    
    print(f"\nOverall Mean Dice: {np.mean(mean_dice_all):.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    validate_model(args.config, args.checkpoint)