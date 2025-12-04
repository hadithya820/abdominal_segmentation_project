"""
V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

Architecture features:
- Residual connections in each stage
- PReLU activations
- Strided convolutions for downsampling
- Memory-optimized for 32GB GPU
- Gradient checkpointing support

Reference: Milletari et al., "V-Net: Fully Convolutional Neural Networks for 
Volumetric Medical Image Segmentation", 3DV 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvBlock3D(nn.Module):
    """Single 3D convolution block with normalization and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.PReLU(out_channels)
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    V-Net style residual block with multiple convolutions
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        n_convs: Number of convolutions (1-3 depending on stage)
        kernel_size: Convolution kernel size
    """
    
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        
        layers = []
        for i in range(n_convs):
            if i == 0:
                layers.append(ConvBlock3D(in_channels, out_channels, kernel_size, padding))
            else:
                layers.append(ConvBlock3D(out_channels, out_channels, kernel_size, padding))
        
        self.convs = nn.Sequential(*layers)
        
        # Skip connection projection if dimensions don't match
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = None
        
        self.activation = nn.PReLU(out_channels)
    
    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.convs(x)
        return self.activation(out + residual)


class DownBlock(nn.Module):
    """Downsampling block using strided convolution followed by residual block"""
    
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=5):
        super().__init__()
        # Strided convolution for downsampling (2x2x2)
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(out_channels)
        )
        self.res_block = ResidualBlock(out_channels, out_channels, n_convs, kernel_size)
    
    def forward(self, x):
        x = self.down_conv(x)
        return self.res_block(x)


class UpBlock(nn.Module):
    """Upsampling block using transposed convolution followed by residual block"""
    
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=5, trilinear=False):
        super().__init__()
        
        if trilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.PReLU(out_channels)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.PReLU(out_channels)
            )
        
        # After concatenation with skip connection
        self.res_block = ResidualBlock(out_channels * 2, out_channels, n_convs, kernel_size)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle padding if sizes don't match
        if x.shape != skip.shape:
            diffZ = skip.size()[2] - x.size()[2]
            diffY = skip.size()[3] - x.size()[3]
            diffX = skip.size()[4] - x.size()[4]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)


class VNet(nn.Module):
    """
    V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    
    Features:
    - Residual connections for better gradient flow
    - PReLU activations
    - Instance normalization for small batch sizes
    - Gradient checkpointing for memory efficiency
    
    Args:
        n_channels: Number of input channels (1 for CT)
        n_classes: Number of output classes
        base_filters: Base number of filters (default 16)
        trilinear: Use trilinear upsampling instead of transposed conv
        use_checkpoint: Enable gradient checkpointing
        deep_supervision: Enable deep supervision
    """
    
    def __init__(self, n_channels=1, n_classes=4, base_filters=16, trilinear=False,
                 use_checkpoint=True, deep_supervision=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint
        self.deep_supervision = deep_supervision
        
        f = base_filters  # 16, 32, 64, 128, 256
        
        # Initial convolution to expand channels
        self.init_conv = nn.Sequential(
            nn.Conv3d(n_channels, f, kernel_size=5, padding=2, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.PReLU(f)
        )
        self.init_res = ResidualBlock(f, f, n_convs=1, kernel_size=5)
        
        # Encoder
        self.down1 = DownBlock(f, f*2, n_convs=2, kernel_size=5)      # 16->32
        self.down2 = DownBlock(f*2, f*4, n_convs=3, kernel_size=5)    # 32->64
        self.down3 = DownBlock(f*4, f*8, n_convs=3, kernel_size=5)    # 64->128
        self.down4 = DownBlock(f*8, f*16, n_convs=3, kernel_size=5)   # 128->256
        
        # Decoder
        self.up1 = UpBlock(f*16, f*8, n_convs=3, kernel_size=5, trilinear=trilinear)   # 256->128
        self.up2 = UpBlock(f*8, f*4, n_convs=3, kernel_size=5, trilinear=trilinear)    # 128->64
        self.up3 = UpBlock(f*4, f*2, n_convs=2, kernel_size=5, trilinear=trilinear)    # 64->32
        self.up4 = UpBlock(f*2, f, n_convs=1, kernel_size=5, trilinear=trilinear)      # 32->16
        
        # Output
        self.outc = nn.Conv3d(f, n_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds1 = nn.Conv3d(f*8, n_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(f*4, n_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(f*2, n_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x0 = self.init_conv(x)
        x0 = self.init_res(x0)
        
        # Encoder with optional checkpointing
        if self.use_checkpoint and self.training:
            x1 = checkpoint(self.down1, x0, use_reentrant=False)
            x2 = checkpoint(self.down2, x1, use_reentrant=False)
            x3 = checkpoint(self.down3, x2, use_reentrant=False)
            x4 = checkpoint(self.down4, x3, use_reentrant=False)
        else:
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x4 = self.down4(x3)
        
        # Decoder with optional checkpointing
        if self.use_checkpoint and self.training:
            d1 = checkpoint(self.up1, x4, x3, use_reentrant=False)
            d2 = checkpoint(self.up2, d1, x2, use_reentrant=False)
            d3 = checkpoint(self.up3, d2, x1, use_reentrant=False)
            d4 = checkpoint(self.up4, d3, x0, use_reentrant=False)
        else:
            d1 = self.up1(x4, x3)
            d2 = self.up2(d1, x2)
            d3 = self.up3(d2, x1)
            d4 = self.up4(d3, x0)
        
        logits = self.outc(d4)
        
        if self.deep_supervision and self.training:
            ds1_out = F.interpolate(self.ds1(d1), size=x.shape[2:], mode='trilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(d2), size=x.shape[2:], mode='trilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(d3), size=x.shape[2:], mode='trilinear', align_corners=True)
            return logits, ds1_out, ds2_out, ds3_out
        
        return logits
    
    @staticmethod
    def get_memory_usage(input_shape=(1, 1, 128, 128, 64), base_filters=16, device='cuda'):
        """Estimate GPU memory usage"""
        model = VNet(n_channels=1, n_classes=4, base_filters=base_filters, use_checkpoint=False)
        model = model.to(device)
        model.train()
        
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(input_shape).to(device)
        
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        del model, x, y
        torch.cuda.empty_cache()
        
        return {
            'peak_memory_gb': peak_memory,
            'model_size_mb': model_size,
            'input_shape': input_shape
        }


# Test the model
if __name__ == "__main__":
    print("Testing V-Net...")
    
    # Create model
    model = VNet(n_channels=1, n_classes=4, base_filters=16, trilinear=False, 
                 use_checkpoint=True, deep_supervision=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Test input shape: (batch, channels, depth, height, width)
    dummy_input = torch.randn(2, 1, 64, 128, 128).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.train()
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Test deep supervision
    print("\n--- Testing Deep Supervision ---")
    model_ds = VNet(n_channels=1, n_classes=4, base_filters=16, deep_supervision=True)
    model_ds = model_ds.to(device)
    model_ds.train()
    
    outputs = model_ds(dummy_input)
    print(f"Main output shape: {outputs[0].shape}")
    print(f"DS outputs: {[o.shape for o in outputs[1:]]}")
