"""
Segmentation Models Package

Available models:
- UNet2D: Standard 2D U-Net for slice-based segmentation
- AttentionUNet2D: 2D U-Net with attention gates
- UNet3D: 3D U-Net for volumetric segmentation
- VNet: V-Net with residual connections
- SegResNet: Memory-efficient 3D segmentation network
- SegResNetLite: Lighter version of SegResNet
"""

from .unet_2d import UNet2D
from .attention_unet_2d import AttentionUNet2D
from .unet_3d import UNet3D
from .vnet import VNet
from .segresnet import SegResNet, SegResNetLite

__all__ = [
    'UNet2D',
    'AttentionUNet2D', 
    'UNet3D',
    'VNet',
    'SegResNet',
    'SegResNetLite'
]


def get_model(model_name: str, **kwargs):
    """
    Factory function to create models by name
    
    Args:
        model_name: Name of the model (case-insensitive)
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    
    Examples:
        >>> model = get_model('unet3d', n_channels=1, n_classes=4)
        >>> model = get_model('vnet', base_filters=16)
    """
    model_name = model_name.lower()
    
    models = {
        'unet2d': UNet2D,
        'unet_2d': UNet2D,
        'attentionunet2d': AttentionUNet2D,
        'attention_unet_2d': AttentionUNet2D,
        'unet3d': UNet3D,
        'unet_3d': UNet3D,
        'vnet': VNet,
        'v_net': VNet,
        'segresnet': SegResNet,
        'seg_res_net': SegResNet,
        'segresnet_lite': SegResNetLite,
        'segresnetlite': SegResNetLite,
    }
    
    if model_name not in models:
        available = ', '.join(sorted(set(models.keys())))
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return models[model_name](**kwargs)
