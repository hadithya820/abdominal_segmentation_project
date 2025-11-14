import json
import random
from pathlib import Path
import numpy as np

def create_data_splits(processed_dir, output_dir, 
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       seed=42):
    """
    Create train/val/test splits
    
    Args:
        processed_dir: Directory with processed data
        output_dir: Directory to save split files
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all processed cases
    ct_files = sorted(list(processed_dir.glob("*_ct.npy")))
    case_ids = [f.stem.replace("_ct", "") for f in ct_files]
    
    print(f"Total cases: {len(case_ids)}")
    
    # Shuffle
    random.shuffle(case_ids)
    
    # Calculate split indices
    n_total = len(case_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Create splits
    train_cases = case_ids[:n_train]
    val_cases = case_ids[n_train:n_train + n_val]
    test_cases = case_ids[n_train + n_val:]
    
    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")
    
    # Save splits
    splits = {
        'train': train_cases,
        'val': val_cases,
        'test': test_cases
    }
    
    for split_name, cases in splits.items():
        output_file = output_dir / f"{split_name}_cases.json"
        with open(output_file, 'w') as f:
            json.dump(cases, f, indent=2)
        print(f"Saved {split_name} split: {output_file}")
    
    # Save complete splits info
    splits_info = {
        'total_cases': n_total,
        'train_count': len(train_cases),
        'val_count': len(val_cases),
        'test_count': len(test_cases),
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'seed': seed
    }
    
    with open(output_dir / "splits_info.json", 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    return splits


if __name__ == "__main__":
    splits = create_data_splits(
        processed_dir="../../data/processed",
        output_dir="../../data/splits",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )