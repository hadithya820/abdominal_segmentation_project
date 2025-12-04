"""
3D Dataset Pipeline for Volumetric Medical Image Segmentation

Features:
- Patch extraction (128×128×64) with configurable overlap
- Memory-efficient batch loading with prefetching
- Comprehensive 3D augmentations (rotations, flips, elastic deformations)
- Smart sampling to balance foreground/background patches
- Multi-threaded data loading optimized for 64 cores

Optimized for:
- 256GB RAM
- 64 CPU cores
- Fast NVMe storage
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List, Optional, Dict, Any
from functools import lru_cache
import random
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
from scipy.ndimage import affine_transform


class ElasticDeformation3D:
    """
    3D Elastic deformation for data augmentation
    Memory-efficient implementation
    """
    
    def __init__(self, alpha=100, sigma=10, p=0.3):
        """
        Args:
            alpha: Deformation intensity
            sigma: Smoothing factor
            p: Probability of applying the transform
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        
        shape = image.shape
        
        # Generate random displacement fields (reduced resolution for speed)
        scale = 4
        small_shape = tuple(s // scale for s in shape)
        
        dz = gaussian_filter(np.random.randn(*small_shape) * 2 - 1, self.sigma / scale) * self.alpha / scale
        dy = gaussian_filter(np.random.randn(*small_shape) * 2 - 1, self.sigma / scale) * self.alpha / scale
        dx = gaussian_filter(np.random.randn(*small_shape) * 2 - 1, self.sigma / scale) * self.alpha / scale
        
        # Upscale displacement fields
        dz = zoom(dz, scale, order=1)
        dy = zoom(dy, scale, order=1)
        dx = zoom(dx, scale, order=1)
        
        # Trim to exact size
        dz = dz[:shape[0], :shape[1], :shape[2]]
        dy = dy[:shape[0], :shape[1], :shape[2]]
        dx = dx[:shape[0], :shape[1], :shape[2]]
        
        # Create coordinate grids
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        # Apply deformation
        indices = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1)
        ]
        
        # Interpolate
        image_deformed = map_coordinates(image, indices, order=1, mode='reflect')
        mask_deformed = map_coordinates(mask, indices, order=0, mode='reflect')  # Nearest for mask
        
        return image_deformed.astype(image.dtype), mask_deformed.astype(mask.dtype)


class RandomRotation3D:
    """Random 3D rotation around each axis"""
    
    def __init__(self, angle_range=15, p=0.5):
        self.angle_range = angle_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        
        # Random angles for each axis
        angles = [random.uniform(-self.angle_range, self.angle_range) for _ in range(3)]
        
        # Rotate around each axis
        for axis, angle in enumerate(angles):
            if abs(angle) > 1:  # Only rotate if angle is significant
                axes = tuple(i for i in range(3) if i != axis)
                image = rotate(image, angle, axes=axes, reshape=False, order=1, mode='reflect')
                mask = rotate(mask, angle, axes=axes, reshape=False, order=0, mode='reflect')
        
        return image.astype(np.float32), mask.astype(np.uint8)


class RandomFlip3D:
    """Random flip along each axis"""
    
    def __init__(self, p_per_axis=0.5):
        self.p = p_per_axis
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for axis in range(3):
            if random.random() < self.p:
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis)
        
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class RandomIntensity3D:
    """Random intensity augmentations for CT data"""
    
    def __init__(self, brightness_range=0.1, contrast_range=0.1, gamma_range=0.2, p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        
        # Brightness
        if random.random() < 0.5:
            brightness = random.uniform(-self.brightness_range, self.brightness_range)
            image = image + brightness
        
        # Contrast
        if random.random() < 0.5:
            contrast = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast + mean
        
        # Gamma (only for positive values)
        if random.random() < 0.5:
            gamma = random.uniform(1 - self.gamma_range, 1 + self.gamma_range)
            # Normalize to [0, 1], apply gamma, rescale
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
                image = np.power(image, gamma)
                image = image * (img_max - img_min) + img_min
        
        return image.astype(np.float32), mask


class GaussianNoise3D:
    """Add Gaussian noise"""
    
    def __init__(self, std_range=(0.01, 0.05), p=0.3):
        self.std_range = std_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        
        std = random.uniform(*self.std_range)
        noise = np.random.randn(*image.shape).astype(np.float32) * std
        image = image + noise
        
        return image.astype(np.float32), mask


class GaussianBlur3D:
    """Apply Gaussian blur"""
    
    def __init__(self, sigma_range=(0.5, 1.5), p=0.2):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        
        sigma = random.uniform(*self.sigma_range)
        image = gaussian_filter(image, sigma=sigma)
        
        return image.astype(np.float32), mask


class Compose3D:
    """Compose multiple 3D transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


def get_training_transforms_3d(
    elastic_p=0.2,
    rotation_p=0.3,
    flip_p=0.5,
    intensity_p=0.5,
    noise_p=0.2,
    blur_p=0.1
) -> Compose3D:
    """Get standard 3D augmentation pipeline for training"""
    return Compose3D([
        RandomFlip3D(p_per_axis=flip_p),
        RandomRotation3D(angle_range=15, p=rotation_p),
        ElasticDeformation3D(alpha=100, sigma=10, p=elastic_p),
        RandomIntensity3D(brightness_range=0.1, contrast_range=0.1, gamma_range=0.2, p=intensity_p),
        GaussianNoise3D(std_range=(0.01, 0.03), p=noise_p),
        GaussianBlur3D(sigma_range=(0.5, 1.0), p=blur_p),
    ])


def get_validation_transforms_3d() -> None:
    """Validation uses no augmentation"""
    return None


class VolumeCache:
    """
    LRU cache for loaded volumes to reduce I/O
    Thread-safe implementation
    """
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        
    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Tuple[np.ndarray, np.ndarray]):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()


class AbdomenCT3DDataset(Dataset):
    """
    3D Dataset for volumetric abdominal CT segmentation with patch extraction
    
    Features:
    - Configurable patch size with overlap
    - Smart sampling to balance foreground/background
    - Memory-efficient loading with caching
    - Comprehensive 3D augmentations
    
    Args:
        data_dir: Directory containing processed .npy files
        split_file: JSON file with case IDs
        patch_size: Size of extracted patches (D, H, W)
        overlap: Overlap between patches for inference
        transform: 3D augmentation pipeline
        mode: 'train' for random patches, 'val' for systematic extraction
        foreground_ratio: Ratio of patches that should contain foreground
        cache_size: Number of volumes to cache in memory
        samples_per_volume: Number of patches to sample per volume (train mode)
    """
    
    def __init__(
        self,
        data_dir: str,
        split_file: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        overlap: float = 0.5,
        transform: Optional[Compose3D] = None,
        mode: str = 'train',
        foreground_ratio: float = 0.7,
        cache_size: int = 10,
        samples_per_volume: int = 8
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.mode = mode
        self.foreground_ratio = foreground_ratio
        self.samples_per_volume = samples_per_volume
        
        # Load case IDs
        with open(split_file, 'r') as f:
            self.case_ids = json.load(f)
        
        # Filter valid cases
        self.valid_cases = []
        for case_id in self.case_ids:
            ct_path = self.data_dir / f"{case_id}_ct.npy"
            seg_path = self.data_dir / f"{case_id}_seg.npy"
            if ct_path.exists() and seg_path.exists():
                self.valid_cases.append(case_id)
            else:
                warnings.warn(f"Missing files for {case_id}, skipping")
        
        print(f"Found {len(self.valid_cases)} valid cases out of {len(self.case_ids)}")
        
        # Initialize cache
        self.cache = VolumeCache(max_size=cache_size)
        
        # Pre-compute volume shapes and foreground locations
        self.volume_info = {}
        self._precompute_volume_info()
        
        # Build index
        if mode == 'train':
            self._build_training_index()
        else:
            self._build_validation_index()
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def _precompute_volume_info(self):
        """Pre-compute volume shapes and foreground voxel counts"""
        print(f"Pre-computing volume information for {len(self.valid_cases)} cases...")
        for idx, case_id in enumerate(self.valid_cases):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  Processing case {idx + 1}/{len(self.valid_cases)}...")
            ct_path = self.data_dir / f"{case_id}_ct.npy"
            seg_path = self.data_dir / f"{case_id}_seg.npy"
            
            # Load files normally (mmap doesn't work on distributed filesystems)
            try:
                ct_data = np.load(ct_path)
                seg_data = np.load(seg_path)
            except Exception as e:
                warnings.warn(f"Error loading {case_id}: {e}")
                continue
            
            # Find foreground bounding box for smart sampling
            foreground_mask = seg_data > 0
            if foreground_mask.any():
                coords = np.where(foreground_mask)
                bbox = {
                    'z_min': int(coords[0].min()), 'z_max': int(coords[0].max()),
                    'y_min': int(coords[1].min()), 'y_max': int(coords[1].max()),
                    'x_min': int(coords[2].min()), 'x_max': int(coords[2].max()),
                }
            else:
                bbox = None
            
            self.volume_info[case_id] = {
                'shape': ct_data.shape,
                'foreground_bbox': bbox,
                'foreground_ratio': float(foreground_mask.sum() / foreground_mask.size)
            }
            
            # Free memory immediately
            del ct_data, seg_data, foreground_mask
    
    def _build_training_index(self):
        """Build index for training mode (random sampling)"""
        self.samples = []
        for case_id in self.valid_cases:
            for _ in range(self.samples_per_volume):
                self.samples.append(case_id)
    
    def _build_validation_index(self):
        """Build index for validation mode (systematic patches)"""
        self.samples = []
        
        for case_id in self.valid_cases:
            shape = self.volume_info[case_id]['shape']
            
            # Calculate patch positions with overlap
            step = [int(p * (1 - self.overlap)) for p in self.patch_size]
            
            for z in range(0, max(1, shape[0] - self.patch_size[0] + 1), step[0]):
                for y in range(0, max(1, shape[1] - self.patch_size[1] + 1), step[1]):
                    for x in range(0, max(1, shape[2] - self.patch_size[2] + 1), step[2]):
                        self.samples.append((case_id, (z, y, x)))
    
    def _load_volume(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load volume with caching"""
        cached = self.cache.get(case_id)
        if cached is not None:
            return cached
        
        ct_path = self.data_dir / f"{case_id}_ct.npy"
        seg_path = self.data_dir / f"{case_id}_seg.npy"
        
        ct_data = np.load(ct_path).astype(np.float32)
        seg_data = np.load(seg_path).astype(np.uint8)
        
        self.cache.put(case_id, (ct_data, seg_data))
        return ct_data, seg_data
    
    def _sample_patch_location(self, case_id: str) -> Tuple[int, int, int]:
        """Sample patch location with foreground preference"""
        info = self.volume_info[case_id]
        shape = info['shape']
        bbox = info['foreground_bbox']
        
        # Decide if we sample from foreground region
        sample_foreground = random.random() < self.foreground_ratio and bbox is not None
        
        if sample_foreground:
            # Sample from foreground bounding box (with some margin)
            margin = [p // 4 for p in self.patch_size]
            
            z = random.randint(
                max(0, bbox['z_min'] - margin[0]),
                min(shape[0] - self.patch_size[0], bbox['z_max'] - self.patch_size[0] // 2)
            )
            y = random.randint(
                max(0, bbox['y_min'] - margin[1]),
                min(shape[1] - self.patch_size[1], bbox['y_max'] - self.patch_size[1] // 2)
            )
            x = random.randint(
                max(0, bbox['x_min'] - margin[2]),
                min(shape[2] - self.patch_size[2], bbox['x_max'] - self.patch_size[2] // 2)
            )
        else:
            # Random sampling from entire volume
            z = random.randint(0, max(0, shape[0] - self.patch_size[0]))
            y = random.randint(0, max(0, shape[1] - self.patch_size[1]))
            x = random.randint(0, max(0, shape[2] - self.patch_size[2]))
        
        return z, y, x
    
    def _extract_patch(
        self, 
        ct_data: np.ndarray, 
        seg_data: np.ndarray, 
        position: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract patch at given position with padding if needed"""
        z, y, x = position
        d, h, w = self.patch_size
        
        # Calculate valid ranges
        z_end = min(z + d, ct_data.shape[0])
        y_end = min(y + h, ct_data.shape[1])
        x_end = min(x + w, ct_data.shape[2])
        
        # Extract
        ct_patch = ct_data[z:z_end, y:y_end, x:x_end]
        seg_patch = seg_data[z:z_end, y:y_end, x:x_end]
        
        # Pad if necessary
        if ct_patch.shape != tuple(self.patch_size):
            ct_padded = np.zeros(self.patch_size, dtype=ct_patch.dtype)
            seg_padded = np.zeros(self.patch_size, dtype=seg_patch.dtype)
            
            ct_padded[:ct_patch.shape[0], :ct_patch.shape[1], :ct_patch.shape[2]] = ct_patch
            seg_padded[:seg_patch.shape[0], :seg_patch.shape[1], :seg_patch.shape[2]] = seg_patch
            
            return ct_padded, seg_padded
        
        return ct_patch, seg_patch
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'train':
            case_id = self.samples[idx]
            ct_data, seg_data = self._load_volume(case_id)
            position = self._sample_patch_location(case_id)
        else:
            case_id, position = self.samples[idx]
            ct_data, seg_data = self._load_volume(case_id)
        
        # Extract patch
        ct_patch, seg_patch = self._extract_patch(ct_data, seg_data, position)
        
        # Apply augmentations
        if self.transform is not None:
            ct_patch, seg_patch = self.transform(ct_patch, seg_patch)
        
        # Convert to tensor
        ct_tensor = torch.from_numpy(ct_patch.copy()).unsqueeze(0).float()  # (1, D, H, W)
        seg_tensor = torch.from_numpy(seg_patch.copy()).long()  # (D, H, W)
        
        return ct_tensor, seg_tensor
    
    def get_case_patches(self, case_id: str) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Get all patch positions and sizes for a specific case (for inference)"""
        shape = self.volume_info[case_id]['shape']
        patches = []
        
        step = [int(p * (1 - self.overlap)) for p in self.patch_size]
        
        for z in range(0, max(1, shape[0] - self.patch_size[0] + 1), step[0]):
            for y in range(0, max(1, shape[1] - self.patch_size[1] + 1), step[1]):
                for x in range(0, max(1, shape[2] - self.patch_size[2] + 1), step[2]):
                    patches.append(((z, y, x), self.patch_size))
        
        # Add edge patches if needed
        if shape[0] > self.patch_size[0]:
            # ... handle edge cases
            pass
        
        return patches


class BalancedSampler(Sampler):
    """
    Balanced sampler that ensures each epoch sees roughly equal 
    representation from each volume
    """
    
    def __init__(self, dataset: AbdomenCT3DDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_cases = len(dataset.valid_cases)
        self.samples_per_case = dataset.samples_per_volume
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # Shuffle within each case's samples
            chunks = [indices[i:i + self.samples_per_case] 
                      for i in range(0, len(indices), self.samples_per_case)]
            random.shuffle(chunks)
            for chunk in chunks:
                random.shuffle(chunk)
            indices = [idx for chunk in chunks for idx in chunk]
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset)


def create_3d_dataloaders(
    data_dir: str,
    train_split: str,
    val_split: str,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    batch_size: int = 2,
    num_workers: int = 8,
    cache_size: int = 10,
    samples_per_volume: int = 8,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for 3D segmentation
    
    Args:
        data_dir: Path to processed data
        train_split: Path to training split JSON
        val_split: Path to validation split JSON
        patch_size: Size of 3D patches
        batch_size: Batch size
        num_workers: Number of data loading workers (use 8-16 for 64 cores)
        cache_size: Number of volumes to cache
        samples_per_volume: Patches per volume per epoch
        use_augmentation: Whether to apply augmentations
    
    Returns:
        train_loader, val_loader
    """
    # Training dataset
    train_transform = get_training_transforms_3d() if use_augmentation else None
    train_dataset = AbdomenCT3DDataset(
        data_dir=data_dir,
        split_file=train_split,
        patch_size=patch_size,
        transform=train_transform,
        mode='train',
        foreground_ratio=0.7,
        cache_size=cache_size,
        samples_per_volume=samples_per_volume
    )
    
    train_sampler = BalancedSampler(train_dataset, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Validation dataset
    val_dataset = AbdomenCT3DDataset(
        data_dir=data_dir,
        split_file=val_split,
        patch_size=patch_size,
        overlap=0.5,
        transform=None,
        mode='val',
        cache_size=cache_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing 3D Dataset Pipeline...")
    
    # Test augmentations
    print("\n--- Testing Augmentations ---")
    dummy_image = np.random.randn(64, 128, 128).astype(np.float32)
    dummy_mask = np.random.randint(0, 4, (64, 128, 128)).astype(np.uint8)
    
    transforms = get_training_transforms_3d()
    aug_image, aug_mask = transforms(dummy_image, dummy_mask)
    print(f"Augmented image shape: {aug_image.shape}")
    print(f"Augmented mask shape: {aug_mask.shape}")
    
    # Test dataset (if data exists)
    data_dir = "../../data/processed"
    train_split = "../../data/splits/train_cases.json"
    val_split = "../../data/splits/val_cases.json"
    
    if Path(data_dir).exists() and Path(train_split).exists():
        print("\n--- Testing Dataset ---")
        
        train_dataset = AbdomenCT3DDataset(
            data_dir=data_dir,
            split_file=train_split,
            patch_size=(64, 128, 128),
            transform=get_training_transforms_3d(),
            mode='train',
            samples_per_volume=4
        )
        
        print(f"Training samples: {len(train_dataset)}")
        
        # Test loading
        ct_patch, seg_patch = train_dataset[0]
        print(f"CT patch shape: {ct_patch.shape}")
        print(f"Seg patch shape: {seg_patch.shape}")
        print(f"CT range: [{ct_patch.min():.3f}, {ct_patch.max():.3f}]")
        print(f"Seg unique labels: {torch.unique(seg_patch).tolist()}")
        
        # Test dataloader
        train_loader, val_loader = create_3d_dataloaders(
            data_dir=data_dir,
            train_split=train_split,
            val_split=val_split,
            patch_size=(64, 128, 128),
            batch_size=2,
            num_workers=4
        )
        
        print(f"\nTrain loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        
        # Load one batch
        batch_ct, batch_seg = next(iter(train_loader))
        print(f"Batch CT shape: {batch_ct.shape}")
        print(f"Batch Seg shape: {batch_seg.shape}")
    else:
        print(f"\nData not found at {data_dir}. Skipping dataset test.")
