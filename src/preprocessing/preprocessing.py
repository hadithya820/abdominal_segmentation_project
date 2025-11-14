import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import zoom
import json

class CTPreprocessor:
    """
    Preprocessing pipeline for abdominal CT scans
    """
    def __init__(self, 
                 hu_window_level=40, 
                 hu_window_width=400,
                 target_spacing=(1.0, 1.0, 1.0),
                 target_organs=[1, 2, 6]):  # Liver, Kidneys, Spleen
        """
        Args:
            hu_window_level: Center of HU window
            hu_window_width: Width of HU window
            target_spacing: Target isotropic spacing (x, y, z) in mm
            target_organs: List of organ labels to keep (AbdomenCT-1K specific)
        """
        self.hu_min = hu_window_level - (hu_window_width / 2)
        self.hu_max = hu_window_level + (hu_window_width / 2)
        self.target_spacing = target_spacing
        self.target_organs = target_organs
    
    def apply_hu_windowing(self, ct_array):
        """
        Apply Hounsfield Unit windowing
        
        Args:
            ct_array: Raw CT array in HU
        Returns:
            Windowed CT array
        """
        ct_windowed = np.clip(ct_array, self.hu_min, self.hu_max)
        return ct_windowed
    
    def normalize_intensity(self, ct_array):
        """
        Normalize intensity to [0, 1]
        
        Args:
            ct_array: Windowed CT array
        Returns:
            Normalized array in [0, 1]
        """
        ct_normalized = (ct_array - self.hu_min) / (self.hu_max - self.hu_min)
        ct_normalized = np.clip(ct_normalized, 0, 1)
        return ct_normalized
    
    def resample_to_isotropic(self, image_array, original_spacing):
        """
        Resample to isotropic spacing using scipy zoom
        
        Args:
            image_array: Input array (CT or segmentation)
            original_spacing: Original voxel spacing (x, y, z)
        Returns:
            Resampled array
        """
        # Calculate zoom factors
        zoom_factors = np.array(original_spacing) / np.array(self.target_spacing)
        
        # Use different interpolation for CT vs segmentation
        # This function should be called separately with appropriate order
        resampled = zoom(image_array, zoom_factors, order=3)  # order=3 for CT, order=0 for seg
        
        return resampled
    
    def filter_organs(self, seg_array):
        """
        Keep only target organs, relabel to [0, 1, 2, 3]
        0: background, 1: liver, 2: kidneys, 3: spleen
        
        Args:
            seg_array: Original segmentation array
        Returns:
            Filtered segmentation with new labels
        """
        new_seg = np.zeros_like(seg_array)
        
        # AbdomenCT-1K labels: 1=liver, 2=kidney_right, 3=kidney_left, 6=spleen
        # Merge kidneys into one class
        new_seg[seg_array == 1] = 1  # Liver
        new_seg[(seg_array == 2) | (seg_array == 3)] = 2  # Both kidneys
        new_seg[seg_array == 6] = 3  # Spleen
        
        return new_seg
    
    def process_case(self, ct_path, seg_path, output_dir, case_id):
        """
        Complete preprocessing pipeline for one case
        
        Args:
            ct_path: Path to CT imaging file
            seg_path: Path to segmentation file
            output_dir: Directory to save processed files
            case_id: Case identifier
        Returns:
            Dictionary with processing info
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        ct_nib = nib.load(ct_path)
        seg_nib = nib.load(seg_path)
        
        ct_data = ct_nib.get_fdata()
        seg_data = seg_nib.get_fdata()
        original_spacing = ct_nib.header.get_zooms()
        
        print(f"Processing {case_id}...")
        print(f"  Original shape: {ct_data.shape}, spacing: {original_spacing}")
        
        # 1. HU Windowing
        ct_windowed = self.apply_hu_windowing(ct_data)
        
        # 2. Normalize intensity
        ct_normalized = self.normalize_intensity(ct_windowed)
        
        # 3. Resample CT to isotropic spacing
        ct_resampled = self.resample_to_isotropic(ct_normalized, original_spacing)
        
        # 4. Resample segmentation (nearest neighbor)
        seg_resampled = zoom(seg_data, 
                            np.array(original_spacing) / np.array(self.target_spacing),
                            order=0)  # Nearest neighbor for labels
        
        # 5. Filter organs
        seg_filtered = self.filter_organs(seg_resampled)
        
        print(f"  Processed shape: {ct_resampled.shape}")
        print(f"  Organ labels: {np.unique(seg_filtered)}")
        
        # Save processed data
        ct_output = output_dir / f"{case_id}_ct.npy"
        seg_output = output_dir / f"{case_id}_seg.npy"
        
        np.save(ct_output, ct_resampled.astype(np.float32))
        np.save(seg_output, seg_filtered.astype(np.uint8))
        
        # Save metadata - convert NumPy types to Python native types for JSON serialization
        metadata = {
            'case_id': str(case_id),
            'original_shape': [int(x) for x in ct_data.shape],
            'processed_shape': [int(x) for x in ct_resampled.shape],
            'original_spacing': [float(x) for x in original_spacing],
            'target_spacing': [float(x) for x in self.target_spacing],
            'organ_labels': [int(x) for x in np.unique(seg_filtered)]
        }
        
        metadata_output = output_dir / f"{case_id}_metadata.json"
        with open(metadata_output, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata


# Usage example
if __name__ == "__main__":
    preprocessor = CTPreprocessor()
    
    # Test on one case
    data_dir = Path("../../data/raw/AbdomenCT-1K")
    output_dir = Path("../../data/processed")
    
    case_dir = data_dir / "Case_00000"
    ct_path = case_dir / "imaging.nii.gz"
    seg_path = case_dir / "segmentation.nii.gz"
    
    metadata = preprocessor.process_case(ct_path, seg_path, output_dir, "Case_00000")
    print("\nProcessing complete!")
    print(f"Metadata: {metadata}")