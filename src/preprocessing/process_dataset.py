import argparse
from pathlib import Path
from preprocessing import CTPreprocessor
from tqdm import tqdm
import json
import re
from multiprocessing import Pool, cpu_count

def process_single_case(args):
    """Helper function for multiprocessing"""
    ct_path, seg_path, case_id, output_dir, preprocessor_params = args
    try:
        preprocessor = CTPreprocessor(**preprocessor_params)
        metadata = preprocessor.process_case(ct_path, seg_path, output_dir, case_id)
        return metadata, None
    except Exception as e:
        return None, (case_id, str(e))


def process_full_dataset(input_dir, output_dir, num_cases=None, num_workers=None):
    """
    Process all cases in the dataset
    
    Args:
        input_dir: Directory containing raw data (supports both AbdomenCT-1K and Subtask1 formats)
        output_dir: Directory to save processed data
        num_cases: Number of cases to process (None = all)
        num_workers: Number of parallel workers (None = use all CPU cores)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect dataset format
    if (input_dir / "Subtask1").exists():
        # Subtask1 format: TrainImage/train_XXXX_0000.nii.gz, TrainMask/train_XXXX.nii.gz
        print("Detected Subtask1 dataset format")
        image_dir = input_dir / "Subtask1" / "TrainImage"
        mask_dir = input_dir / "Subtask1" / "TrainMask"
        
        # Get all training images
        image_files = sorted(list(image_dir.glob("train_*_0000.nii.gz")))
        
        # Extract case IDs and create pairs
        case_pairs = []
        for image_file in image_files:
            # Extract case number from train_0001_0000.nii.gz -> 0001
            match = re.search(r'train_(\d+)_0000', image_file.name)
            if match:
                case_num = match.group(1)
                mask_file = mask_dir / f"train_{case_num}.nii.gz"
                
                if mask_file.exists():
                    case_pairs.append((image_file, mask_file, f"train_{case_num}"))
                else:
                    print(f"Warning: Missing mask for {image_file.name}")
        
        print(f"Found {len(case_pairs)} image-mask pairs")
        
    else:
        # AbdomenCT-1K format: Case_XXXXX/imaging.nii.gz, Case_XXXXX/segmentation.nii.gz
        print("Detected AbdomenCT-1K dataset format")
        case_dirs = sorted(list(input_dir.glob("Case_*")))
        
        case_pairs = []
        for case_dir in case_dirs:
            ct_path = case_dir / "imaging.nii.gz"
            seg_path = case_dir / "segmentation.nii.gz"
            
            if ct_path.exists() and seg_path.exists():
                case_pairs.append((ct_path, seg_path, case_dir.name))
            else:
                print(f"Warning: Missing files in {case_dir.name}")
        
        print(f"Found {len(case_pairs)} cases")
    
    # Limit number of cases if specified
    if num_cases:
        case_pairs = case_pairs[:num_cases]
        print(f"Processing first {num_cases} cases")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    print(f"Using {num_workers} parallel workers")
    
    # Initialize preprocessor parameters
    preprocessor_params = {
        'hu_window_level': 40,
        'hu_window_width': 400,
        'target_spacing': (1.0, 1.0, 1.0)
    }
    
    # Prepare arguments for multiprocessing
    process_args = [
        (ct_path, seg_path, case_id, output_dir, preprocessor_params)
        for ct_path, seg_path, case_id in case_pairs
    ]
    
    # Process cases in parallel
    all_metadata = []
    failed_cases = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_case, process_args),
            total=len(process_args),
            desc="Processing cases"
        ))
    
    # Separate successful and failed cases
    for metadata, error in results:
        if metadata is not None:
            all_metadata.append(metadata)
        if error is not None:
            case_id, error_msg = error
            print(f"Error processing {case_id}: {error_msg}")
            failed_cases.append(case_id)
    
    # Save summary
    summary = {
        'total_cases': len(case_pairs),
        'successful': len(all_metadata),
        'failed': len(failed_cases),
        'failed_cases': failed_cases
    }
    
    with open(output_dir / "processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"{'='*50}")
    print(f"Successful: {len(all_metadata)}/{len(case_pairs)}")
    print(f"Failed: {len(failed_cases)}")
    if failed_cases:
        print(f"Failed cases: {failed_cases}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CT dataset (supports AbdomenCT-1K and Subtask1 formats)")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Path to raw data directory (e.g., /dbfs/tmp/html_output/data)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save processed data")
    parser.add_argument("--num_cases", type=int, default=None,
                       help="Number of cases to process (default: all)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: use all CPU cores - 1)")
    
    args = parser.parse_args()
    
    process_full_dataset(args.input_dir, args.output_dir, args.num_cases, args.num_workers)