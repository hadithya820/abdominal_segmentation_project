import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import AbdomenCTDataset, get_validation_augmentation
import argparse
import json
from pathlib import Path
import pandas as pd

def get_model(model_name):
    if model_name.lower() == "unet2d":
        from models.unet_2d import UNet2D
        return UNet2D
    elif model_name.lower() == "attentionunet2d":
        from models.attention_unet_2d import AttentionUNet2D
        return AttentionUNet2D
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def validate_and_report(config_path, checkpoint_path, save_csv=None):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ModelClass = get_model(config['model_name'])
    model = ModelClass(
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        bilinear=config.get('bilinear', False)
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Dataset
    val_dataset = AbdomenCTDataset(
        data_dir=config['data_dir'],
        split_file=config['val_split'],
        transform=get_validation_augmentation(),
        cache_data=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    organ_names = ['Liver', 'Kidneys', 'Spleen']
    organ_classes = [1, 2, 3]  # Adjust this if your class mapping differs

    dice_scores = {org: [] for org in organ_names}

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            for i, org in enumerate(organ_names):
                cls_idx = organ_classes[i]
                pred_bin = (preds == cls_idx).float()
                mask_bin = (masks == cls_idx).float()
                intersection = (pred_bin * mask_bin).sum(dim=[1,2])
                denom = pred_bin.sum(dim=[1,2]) + mask_bin.sum(dim=[1,2])
                dice = (2 * intersection + 1e-8) / (denom + 1e-8)
                dice_scores[org].extend(dice.cpu().numpy().tolist())

    print("\n===== PER-ORGAN DICE REPORT =====")
    summary = []
    all_means = []
    for org in organ_names:
        mean = np.mean(dice_scores[org])
        std = np.std(dice_scores[org])
        print(f"{org:>10}: {mean:.4f} ± {std:.4f}")
        summary.append({'Organ': org, 'Mean Dice': mean, 'Std Dice': std})
        all_means.append(mean)
    overall = np.mean(all_means)
    print(f"\nOverall Mean Dice: {overall:.4f}")
    print("="*40)
    if save_csv:
        pd.DataFrame(summary).to_csv(save_csv, index=False)
        print(f"Saved results to {save_csv}")
    return {org: dice_scores[org] for org in organ_names}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate U-Net/Attention U-Net and report per-organ Dice scores")
    parser.add_argument('--config', type=str, required=True, help="Path to config .json")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument('--save_csv', type=str, default=None, help="Optional path to save Dice summary as CSV")
    args = parser.parse_args()

    validate_and_report(args.config, args.checkpoint, args.save_csv)
