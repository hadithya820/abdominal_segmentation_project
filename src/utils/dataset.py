import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AbdomenCTDataset(Dataset):
    """
    PyTorch Dataset for 2D slice-based abdominal CT segmentation
    """
    def __init__(self, 
                 data_dir,
                 split_file,
                 slice_axis=2,  # Which axis to slice (0, 1, or 2)
                 transform=None,
                 cache_data=False):
        """
        Args:
            data_dir: Directory containing processed .npy files
            split_file: JSON file with case IDs for this split
            slice_axis: Axis along which to extract 2D slices (default: 2 = axial)
            transform: Albumentations transform pipeline
            cache_data: Whether to cache all data in memory
        """
        self.data_dir = Path(data_dir)
        self.slice_axis = slice_axis
        self.transform = transform
        self.cache_data = cache_data
        
        # Load case IDs
        with open(split_file, 'r') as f:
            self.case_ids = json.load(f)
        
        # Build slice index
        self.slices = []
        self.cache = {} if cache_data else None
        
        print(f"Building slice index from {len(self.case_ids)} cases...")
        for case_id in self.case_ids:
            ct_path = self.data_dir / f"{case_id}_ct.npy"
            seg_path = self.data_dir / f"{case_id}_seg.npy"
            
            if not ct_path.exists() or not seg_path.exists():
                print(f"Warning: Missing files for {case_id}, skipping")
                continue
            
            # Load to get shape
            ct_data = np.load(ct_path)
            n_slices = ct_data.shape[self.slice_axis]
            
            # Cache if requested
            if self.cache_data:
                seg_data = np.load(seg_path)
                self.cache[case_id] = (ct_data, seg_data)
            
            # Add all slices from this case
            for slice_idx in range(n_slices):
                self.slices.append((case_id, slice_idx))
        
        print(f"Dataset initialized with {len(self.slices)} slices from {len(self.case_ids)} cases")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        case_id, slice_idx = self.slices[idx]
        
        # Load or retrieve from cache
        if self.cache_data:
            ct_data, seg_data = self.cache[case_id]
        else:
            ct_path = self.data_dir / f"{case_id}_ct.npy"
            seg_path = self.data_dir / f"{case_id}_seg.npy"
            ct_data = np.load(ct_path)
            seg_data = np.load(seg_path)
        
        # Extract 2D slice
        if self.slice_axis == 0:
            ct_slice = ct_data[slice_idx, :, :]
            seg_slice = seg_data[slice_idx, :, :]
        elif self.slice_axis == 1:
            ct_slice = ct_data[:, slice_idx, :]
            seg_slice = seg_data[:, slice_idx, :]
        else:  # axis 2 (axial - default)
            ct_slice = ct_data[:, :, slice_idx]
            seg_slice = seg_data[:, :, slice_idx]
        
        # Convert to float32 and uint8
        ct_slice = ct_slice.astype(np.float32)
        seg_slice = seg_slice.astype(np.uint8)
        
        # Apply augmentation
        if self.transform:
            transformed = self.transform(image=ct_slice, mask=seg_slice)
            ct_slice = transformed['image']
            seg_slice = transformed['mask']
        else:
            # Convert to tensor manually if no transform
            ct_slice = torch.from_numpy(ct_slice).unsqueeze(0)  # Add channel dim
            seg_slice = torch.from_numpy(seg_slice).long()
        
        return ct_slice, seg_slice


def get_training_augmentation(image_size=512):
    """
    Get albumentations transform for training
    
    Args:
        image_size: Target size for all images (default: 512)
    """
    transform = A.Compose([
        # Resize to consistent dimensions
        A.Resize(height=image_size, width=image_size),
        
        # Geometric transforms
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        
        # Intensity transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        
        # Normalization and tensor conversion
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),
        ToTensorV2()
    ])
    return transform


def get_validation_augmentation(image_size=512):
    """
    Get albumentations transform for validation/testing
    
    Args:
        image_size: Target size for all images (default: 512)
    """
    transform = A.Compose([
        # Resize to consistent dimensions
        A.Resize(height=image_size, width=image_size),
        
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),
        ToTensorV2()
    ])
    return transform


# Test the dataset
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    # Create dataset
    train_dataset = AbdomenCTDataset(
        data_dir="../../data/processed",
        split_file="../../data/splits/train_cases.json",
        transform=get_training_augmentation(),
        cache_data=False
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # Test loading
    print("Testing data loading...")
    ct_batch, seg_batch = next(iter(train_loader))
    print(f"CT batch shape: {ct_batch.shape}")
    print(f"Seg batch shape: {seg_batch.shape}")
    print(f"CT range: [{ct_batch.min():.3f}, {ct_batch.max():.3f}]")
    print(f"Seg unique labels: {torch.unique(seg_batch)}")
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        # Denormalize for visualization
        ct_vis = ct_batch[i, 0].numpy() * 0.5 + 0.5
        seg_vis = seg_batch[i].numpy()
        
        axes[0, i].imshow(ct_vis, cmap='gray')
        axes[0, i].set_title(f'CT Slice {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(seg_vis, cmap='jet')
        axes[1, i].set_title(f'Segmentation {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('../../results/visualizations/dataloader_test.png', dpi=150)
    print("Visualization saved!")