#!/usr/bin/env python3
"""
Preprocess only the test cases (55 cases) for evaluation.

This script reads the test_cases.json and preprocesses only those cases
from the raw Subtask1 data.

Usage:
    python preprocess_test_cases.py \
        --input_dir /localscratch/$USER/flare_data/Subtask1 \
        --output_dir ~/CSE803_Project/abdominal_segmentation_project/data/processed \
        --splits_file ~/CSE803_Project/abdominal_segmentation_project/data/splits/test_cases.json
"""

import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage


class CTPreprocessor:
    """Preprocessor for CT scans - matches original project preprocessing."""
    
    def __init__(self, hu_window_level=40, hu_window_width=400, target_spacing=(1.0, 1.0, 1.0)):
        self.hu_window_level = hu_window_level
        self.hu_window_width = hu_window_width
        self.target_spacing = target_spacing
    
    def apply_hu_window(self, ct_array):
        """Apply HU windowing for abdominal soft tissue."""
        min_hu = self.hu_window_level - self.hu_window_width // 2  # -160
        max_hu = self.hu_window_level + self.hu_window_width // 2  # 240
        
        ct_windowed = np.clip(ct_array, min_hu, max_hu)
        ct_normalized = (ct_windowed - min_hu) / (max_hu - min_hu)
        
        return ct_normalized.astype(np.float32)
    
    def resample_volume(self, volume, original_spacing, is_mask=False):
        """Resample volume to target spacing."""
        # Calculate zoom factors
        zoom_factors = [
            original_spacing[i] / self.target_spacing[i] 
            for i in range(3)
        ]
        
        # Use nearest neighbor for masks, linear for CT
        order = 0 if is_mask else 1
        
        resampled = ndimage.zoom(volume, zoom_factors, order=order)
        
        return resampled
    
    def filter_organs(self, mask):
        """
        Filter and remap organ labels.
        Original labels vary by dataset, we want:
            0: Background
            1: Liver
            2: Kidneys (merge left+right if separate)
            3: Spleen
        """
        # Create new mask
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Common FLARE21 label mapping:
        # 1: Liver, 2: Kidney, 3: Spleen, 4: Pancreas
        # We only keep liver, kidney, spleen
        
        new_mask[mask == 1] = 1  # Liver
        new_mask[mask == 2] = 2  # Kidneys
        new_mask[mask == 3] = 3  # Spleen
        
        return new_mask
    
    def process_case(self, ct_path, mask_path, output_dir, case_id):
        """Process a single case."""
        # Load NIfTI files
        ct_nii = nib.load(str(ct_path))
        mask_nii = nib.load(str(mask_path))
        
        ct_array = ct_nii.get_fdata()
        mask_array = mask_nii.get_fdata().astype(np.uint8)
        
        # Get original spacing from header
        original_spacing = ct_nii.header.get_zooms()[:3]
        
        # Apply HU windowing
        ct_processed = self.apply_hu_window(ct_array)
        
        # Resample to target spacing
        ct_resampled = self.resample_volume(ct_processed, original_spacing, is_mask=False)
        mask_resampled = self.resample_volume(mask_array, original_spacing, is_mask=True)
        
        # Filter organs (keep only liver, kidneys, spleen)
        mask_filtered = self.filter_organs(mask_resampled)
        
        # Save as numpy arrays
        output_dir = Path(output_dir)
        np.save(output_dir / f"{case_id}_ct.npy", ct_resampled)
        np.save(output_dir / f"{case_id}_mask.npy", mask_filtered)
        
        # Save metadata
        metadata = {
            'case_id': case_id,
            'original_shape': list(ct_array.shape),
            'processed_shape': list(ct_resampled.shape),
            'original_spacing': [float(x) for x in original_spacing],
            'target_spacing': list(self.target_spacing),
            'hu_window': {'level': self.hu_window_level, 'width': self.hu_window_width}
        }
        
        with open(output_dir / f"{case_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess test cases only')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to Subtask1 directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output processed data')
    parser.add_argument('--splits_file', type=str, required=True,
                        help='Path to test_cases.json')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test case IDs
    with open(args.splits_file, 'r') as f:
        test_cases = json.load(f)
    
    print(f"Found {len(test_cases)} test cases to process")
    
    # Initialize preprocessor
    preprocessor = CTPreprocessor(
        hu_window_level=40,
        hu_window_width=400,
        target_spacing=(1.0, 1.0, 1.0)
    )
    
    # Process each test case
    success_count = 0
    failed_cases = []
    
    for case_id in tqdm(test_cases, desc="Preprocessing test cases"):
        # Build file paths
        # case_id is like "train_0186"
        case_num = case_id.split('_')[1]  # Extract "0186"
        
        ct_path = input_dir / "TrainImage" / f"{case_id}_0000.nii.gz"
        mask_path = input_dir / "TrainMask" / f"{case_id}.nii.gz"
        
        if not ct_path.exists():
            print(f"\nWarning: CT not found: {ct_path}")
            failed_cases.append(case_id)
            continue
        
        if not mask_path.exists():
            print(f"\nWarning: Mask not found: {mask_path}")
            failed_cases.append(case_id)
            continue
        
        try:
            preprocessor.process_case(ct_path, mask_path, output_dir, case_id)
            success_count += 1
        except Exception as e:
            print(f"\nError processing {case_id}: {e}")
            failed_cases.append(case_id)
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Successfully processed: {success_count}/{len(test_cases)} cases")
    
    if failed_cases:
        print(f"Failed cases: {failed_cases}")
    
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
