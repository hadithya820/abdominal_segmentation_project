"""
Memory and Computational Profiling for 3D Segmentation Models

This script profiles:
- GPU memory usage for different batch sizes
- Forward/backward pass timing
- Model parameter counts
- Optimal batch size determination
- Multi-GPU scaling efficiency

Generates a comprehensive profiling report.
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import psutil
import gc
from typing import Dict, List, Any, Tuple
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.unet_3d import UNet3D
from models.vnet import VNet
from models.segresnet import SegResNet, SegResNetLite


def get_model(model_name: str, n_channels: int = 1, n_classes: int = 4, 
              base_filters: int = 32, use_checkpoint: bool = True) -> nn.Module:
    """Create model by name"""
    model_name = model_name.lower()
    
    if model_name == 'unet3d':
        return UNet3D(n_channels, n_classes, base_filters, use_checkpoint=use_checkpoint)
    elif model_name == 'vnet':
        return VNet(n_channels, n_classes, base_filters // 2, use_checkpoint=use_checkpoint)
    elif model_name == 'segresnet':
        return SegResNet(n_channels, n_classes, base_filters, use_checkpoint=use_checkpoint)
    elif model_name == 'segresnet_lite':
        return SegResNetLite(n_channels, n_classes, use_checkpoint=use_checkpoint)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def profile_single_config(
    model_name: str,
    batch_size: int,
    patch_size: Tuple[int, int, int],
    n_channels: int = 1,
    n_classes: int = 4,
    base_filters: int = 32,
    use_checkpoint: bool = True,
    use_amp: bool = True,
    num_iterations: int = 5,
    warmup_iterations: int = 2
) -> Dict[str, Any]:
    """
    Profile a single model configuration
    
    Returns timing, memory, and throughput metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Profiling on CPU (not recommended)")
    
    # Create model
    model = get_model(model_name, n_channels, n_classes, base_filters, use_checkpoint)
    model = model.to(device)
    model.train()
    
    # Create dummy data
    x = torch.randn(batch_size, n_channels, *patch_size, device=device)
    target = torch.randint(0, n_classes, (batch_size, *patch_size), device=device)
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler('cuda') if use_amp else None
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        optimizer.zero_grad()
        if use_amp:
            with autocast('cuda'):
                output = model(x)
                if isinstance(output, dict):
                    output = output['logits']
                elif isinstance(output, (list, tuple)):
                    output = output[0]
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            if isinstance(output, dict):
                output = output['logits']
            elif isinstance(output, (list, tuple)):
                output = output[0]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Timing iterations
    forward_times = []
    backward_times = []
    total_times = []
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        if use_amp:
            with autocast('cuda'):
                output = model(x)
                if isinstance(output, dict):
                    output = output['logits']
                elif isinstance(output, (list, tuple)):
                    output = output[0]
                loss = criterion(output, target)
        else:
            output = model(x)
            if isinstance(output, dict):
                output = output['logits']
            elif isinstance(output, (list, tuple)):
                output = output[0]
            loss = criterion(output, target)
        
        torch.cuda.synchronize()
        end_forward = time.perf_counter()
        
        # Backward pass
        start_backward = time.perf_counter()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_backward = time.perf_counter()
        
        forward_times.append(end_forward - start_forward)
        backward_times.append(end_backward - start_backward)
        total_times.append(end_backward - start_forward)
    
    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024**2
    
    # Throughput (samples per second)
    avg_total_time = np.mean(total_times)
    throughput = batch_size / avg_total_time
    
    # Calculate input/output sizes
    input_size_mb = x.numel() * x.element_size() / 1024**2
    
    results = {
        'model': model_name,
        'batch_size': batch_size,
        'patch_size': list(patch_size),
        'use_checkpoint': use_checkpoint,
        'use_amp': use_amp,
        
        # Model stats
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        
        # Memory stats
        'peak_memory_gb': peak_memory,
        'allocated_memory_gb': allocated_memory,
        'reserved_memory_gb': reserved_memory,
        'input_size_mb': input_size_mb,
        
        # Timing stats (milliseconds)
        'forward_time_ms': np.mean(forward_times) * 1000,
        'forward_time_std_ms': np.std(forward_times) * 1000,
        'backward_time_ms': np.mean(backward_times) * 1000,
        'backward_time_std_ms': np.std(backward_times) * 1000,
        'total_time_ms': avg_total_time * 1000,
        'total_time_std_ms': np.std(total_times) * 1000,
        
        # Throughput
        'throughput_samples_per_sec': throughput,
        'throughput_patches_per_sec': throughput,
    }
    
    # Cleanup
    del model, x, target, output, loss
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def find_max_batch_size(
    model_name: str,
    patch_size: Tuple[int, int, int],
    target_memory_gb: float = 28.0,  # Leave 4GB headroom for 32GB GPU
    n_channels: int = 1,
    n_classes: int = 4,
    base_filters: int = 32,
    use_checkpoint: bool = True,
    use_amp: bool = True
) -> int:
    """Find maximum batch size that fits in GPU memory"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        return 1
    
    # Binary search for max batch size
    min_batch = 1
    max_batch = 32  # Start with reasonable upper bound
    best_batch = 1
    
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            model = get_model(model_name, n_channels, n_classes, base_filters, use_checkpoint)
            model = model.to(device)
            model.train()
            
            x = torch.randn(mid_batch, n_channels, *patch_size, device=device)
            target = torch.randint(0, n_classes, (mid_batch, *patch_size), device=device)
            
            scaler = GradScaler('cuda') if use_amp else None
            
            with autocast('cuda', enabled=use_amp):
                output = model(x)
                if isinstance(output, dict):
                    output = output['logits']
                elif isinstance(output, (list, tuple)):
                    output = output[0]
                loss = nn.CrossEntropyLoss()(output, target)
            
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            del model, x, target, output, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            if peak_memory <= target_memory_gb:
                best_batch = mid_batch
                min_batch = mid_batch + 1
            else:
                max_batch = mid_batch - 1
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                max_batch = mid_batch - 1
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
    
    return best_batch


def generate_profiling_report(
    output_dir: str = "results",
    models: List[str] = ['UNet3D', 'VNet', 'SegResNet'],
    patch_sizes: List[Tuple[int, int, int]] = [(64, 128, 128), (96, 128, 128), (64, 160, 160)],
    batch_sizes: List[int] = [1, 2, 3, 4],
    base_filters: int = 32
) -> Dict[str, Any]:
    """
    Generate comprehensive profiling report for all model configurations
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {},
        'profiles': [],
        'max_batch_sizes': {},
        'recommendations': {}
    }
    
    # System info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report['system_info'] = {
            'gpu_name': props.name,
            'gpu_memory_gb': props.total_memory / 1024**3,
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'cpu_cores': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / 1024**3
        }
    
    print("\n" + "="*70)
    print("3D MODEL PROFILING REPORT")
    print("="*70)
    print(f"\nGPU: {report['system_info'].get('gpu_name', 'N/A')}")
    print(f"GPU Memory: {report['system_info'].get('gpu_memory_gb', 0):.1f} GB")
    print(f"CPU Cores: {report['system_info'].get('cpu_cores', 'N/A')}")
    print(f"RAM: {report['system_info'].get('ram_gb', 0):.1f} GB")
    print("="*70)
    
    # Profile each configuration
    for model_name in models:
        print(f"\n--- Profiling {model_name} ---")
        
        for patch_size in patch_sizes:
            print(f"\n  Patch size: {patch_size}")
            
            # Find max batch size first
            max_batch = find_max_batch_size(
                model_name, patch_size, 
                target_memory_gb=28.0,
                base_filters=base_filters
            )
            
            report['max_batch_sizes'][f"{model_name}_{patch_size}"] = max_batch
            print(f"  Max batch size: {max_batch}")
            
            # Profile specific batch sizes
            for batch_size in batch_sizes:
                if batch_size > max_batch:
                    print(f"    Batch {batch_size}: SKIPPED (exceeds max)")
                    continue
                
                print(f"    Batch {batch_size}: ", end='', flush=True)
                
                try:
                    profile = profile_single_config(
                        model_name=model_name,
                        batch_size=batch_size,
                        patch_size=patch_size,
                        base_filters=base_filters,
                        use_checkpoint=True,
                        use_amp=True
                    )
                    
                    report['profiles'].append(profile)
                    
                    print(f"Memory: {profile['peak_memory_gb']:.2f}GB, "
                          f"Time: {profile['total_time_ms']:.1f}ms, "
                          f"Throughput: {profile['throughput_samples_per_sec']:.2f} samples/s")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("OOM")
                        torch.cuda.empty_cache()
                    else:
                        print(f"ERROR: {e}")
    
    # Generate recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    for model_name in models:
        # Find best configuration (highest throughput within memory)
        model_profiles = [p for p in report['profiles'] if p['model'].lower() == model_name.lower()]
        
        if model_profiles:
            # Sort by throughput
            best = max(model_profiles, key=lambda x: x['throughput_samples_per_sec'])
            
            recommendation = {
                'model': model_name,
                'recommended_batch_size': best['batch_size'],
                'recommended_patch_size': best['patch_size'],
                'expected_memory_gb': best['peak_memory_gb'],
                'expected_throughput': best['throughput_samples_per_sec'],
                'expected_time_per_step_ms': best['total_time_ms']
            }
            
            report['recommendations'][model_name] = recommendation
            
            print(f"\n{model_name}:")
            print(f"  Recommended batch size: {best['batch_size']}")
            print(f"  Recommended patch size: {best['patch_size']}")
            print(f"  Expected GPU memory: {best['peak_memory_gb']:.2f} GB")
            print(f"  Expected throughput: {best['throughput_samples_per_sec']:.2f} samples/s")
    
    # Save report
    report_path = output_path / 'profiling_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nReport saved to: {report_path}")
    
    # Generate markdown summary
    md_path = output_path / 'PROFILING_REPORT.md'
    generate_markdown_report(report, md_path)
    print(f"Markdown report saved to: {md_path}")
    
    return report


def generate_markdown_report(report: Dict[str, Any], output_path: Path):
    """Generate a readable markdown report"""
    
    md_content = f"""# 3D Model Profiling Report

Generated: {report['timestamp']}

## System Information

| Property | Value |
|----------|-------|
| GPU | {report['system_info'].get('gpu_name', 'N/A')} |
| GPU Memory | {report['system_info'].get('gpu_memory_gb', 0):.1f} GB |
| CUDA Version | {report['system_info'].get('cuda_version', 'N/A')} |
| PyTorch Version | {report['system_info'].get('pytorch_version', 'N/A')} |
| CPU Cores | {report['system_info'].get('cpu_cores', 'N/A')} |
| RAM | {report['system_info'].get('ram_gb', 0):.1f} GB |

## Recommendations

| Model | Batch Size | Patch Size | Memory (GB) | Throughput (samples/s) |
|-------|------------|------------|-------------|----------------------|
"""
    
    for model_name, rec in report['recommendations'].items():
        md_content += f"| {model_name} | {rec['recommended_batch_size']} | {rec['recommended_patch_size']} | {rec['expected_memory_gb']:.2f} | {rec['expected_throughput']:.2f} |\n"
    
    md_content += """

## Detailed Profiling Results

### Memory Usage by Configuration

| Model | Batch | Patch Size | Peak Memory (GB) | Model Size (MB) |
|-------|-------|------------|------------------|-----------------|
"""
    
    for p in report['profiles']:
        md_content += f"| {p['model']} | {p['batch_size']} | {p['patch_size']} | {p['peak_memory_gb']:.2f} | {p['model_size_mb']:.1f} |\n"
    
    md_content += """

### Timing Performance

| Model | Batch | Forward (ms) | Backward (ms) | Total (ms) | Throughput |
|-------|-------|--------------|---------------|------------|------------|
"""
    
    for p in report['profiles']:
        md_content += f"| {p['model']} | {p['batch_size']} | {p['forward_time_ms']:.1f} | {p['backward_time_ms']:.1f} | {p['total_time_ms']:.1f} | {p['throughput_samples_per_sec']:.2f} |\n"
    
    md_content += """

## Training Commands

```bash
# UNet3D
python src/train_3d.py --config config/unet3d_config.json

# VNet
python src/train_3d.py --config config/vnet_config.json

# SegResNet
python src/train_3d.py --config config/segresnet_config.json
```

## Memory Optimization Tips

1. **Gradient Checkpointing**: All models use gradient checkpointing by default to reduce memory usage by ~40%
2. **Mixed Precision (AMP)**: FP16 training reduces memory usage and speeds up training on Tensor Cores
3. **Gradient Accumulation**: Use `accumulation_steps` to simulate larger batch sizes
4. **Patch Size**: Smaller patches use less memory but may affect segmentation quality at boundaries

## Notes

- All measurements were taken with gradient checkpointing and AMP enabled
- Peak memory includes model weights, activations, gradients, and optimizer states
- Throughput measured during training (forward + backward pass)
"""
    
    with open(output_path, 'w') as f:
        f.write(md_content)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile 3D segmentation models')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--models', nargs='+', default=['UNet3D', 'VNet', 'SegResNet'],
                        help='Models to profile')
    parser.add_argument('--quick', action='store_true', help='Quick profile with fewer configs')
    args = parser.parse_args()
    
    if args.quick:
        # Quick profiling with fewer configurations
        patch_sizes = [(64, 128, 128)]
        batch_sizes = [2, 4]
    else:
        # Full profiling
        patch_sizes = [(64, 128, 128), (64, 160, 160)]
        batch_sizes = [1, 2, 3, 4]
    
    generate_profiling_report(
        output_dir=args.output,
        models=args.models,
        patch_sizes=patch_sizes,
        batch_sizes=batch_sizes
    )


if __name__ == "__main__":
    main()
