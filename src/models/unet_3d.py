"""
3D U-Net for Volumetric Medical Image Segmentation

Architecture optimized for:
- 32GB GPU memory
- 128x128x64 patch size
- Gradient checkpointing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv3D(nn.Module):
    """(Conv3D -> BN -> ReLU) x 2 with optional residual connection"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        
        self.activation = nn.LeakyReLU(0.01, inplace=True)
        
        # Residual connection if dimensions match
        if residual and in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = None
    
    def forward(self, x):
        out = self.double_conv(x)
        if self.residual:
            if self.skip is not None:
                x = self.skip(x)
            out = out + x
        return self.activation(out)


class Down3D(nn.Module):
    """Downscaling with strided conv then double conv"""
    
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv = DoubleConv3D(in_channels, out_channels, residual=residual)
    
    def forward(self, x):
        x = self.down_conv(x)
        return self.conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, trilinear=True, residual=False):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, residual=residual)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, residual=residual)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding if sizes don't match
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation.
    
    Optimized for:
    - Memory efficiency with gradient checkpointing
    - Instance normalization for small batch sizes
    - LeakyReLU for stable training
    
    Args:
        n_channels: Number of input channels (1 for CT)
        n_classes: Number of output classes (4: background, liver, kidneys, spleen)
        base_filters: Base number of filters (default 32 for memory efficiency)
        trilinear: Use trilinear upsampling (True) or transposed conv (False)
        use_checkpoint: Enable gradient checkpointing for memory efficiency
        deep_supervision: Enable deep supervision for better gradients
    """
    
    def __init__(self, n_channels=1, n_classes=4, base_filters=32, trilinear=True, 
                 use_checkpoint=True, deep_supervision=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        self.use_checkpoint = use_checkpoint
        self.deep_supervision = deep_supervision
        
        # Feature channels: 32, 64, 128, 256, 512 (reduced from 2D for memory)
        f = base_filters
        
        # Encoder
        self.inc = DoubleConv3D(n_channels, f, residual=True)
        self.down1 = Down3D(f, f*2, residual=True)
        self.down2 = Down3D(f*2, f*4, residual=True)
        self.down3 = Down3D(f*4, f*8, residual=True)
        
        factor = 2 if trilinear else 1
        self.down4 = Down3D(f*8, f*16 // factor, residual=True)
        
        # Decoder
        self.up1 = Up3D(f*16, f*8 // factor, trilinear, residual=True)
        self.up2 = Up3D(f*8, f*4 // factor, trilinear, residual=True)
        self.up3 = Up3D(f*4, f*2 // factor, trilinear, residual=True)
        self.up4 = Up3D(f*2, f, trilinear, residual=True)
        
        self.outc = nn.Conv3d(f, n_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds1 = nn.Conv3d(f*8 // factor, n_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(f*4 // factor, n_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(f*2 // factor, n_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _forward_encoder(self, x):
        x1 = self.inc(x)
        
        if self.use_checkpoint and self.training:
            x2 = checkpoint(self.down1, x1, use_reentrant=False)
            x3 = checkpoint(self.down2, x2, use_reentrant=False)
            x4 = checkpoint(self.down3, x3, use_reentrant=False)
            x5 = checkpoint(self.down4, x4, use_reentrant=False)
        else:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        
        return x1, x2, x3, x4, x5
    
    def _forward_decoder(self, x1, x2, x3, x4, x5):
        if self.use_checkpoint and self.training:
            d1 = checkpoint(self.up1, x5, x4, use_reentrant=False)
            d2 = checkpoint(self.up2, d1, x3, use_reentrant=False)
            d3 = checkpoint(self.up3, d2, x2, use_reentrant=False)
            d4 = checkpoint(self.up4, d3, x1, use_reentrant=False)
        else:
            d1 = self.up1(x5, x4)
            d2 = self.up2(d1, x3)
            d3 = self.up3(d2, x2)
            d4 = self.up4(d3, x1)
        
        return d1, d2, d3, d4
    
    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self._forward_encoder(x)
        
        # Decoder
        d1, d2, d3, d4 = self._forward_decoder(x1, x2, x3, x4, x5)
        
        logits = self.outc(d4)
        
        if self.deep_supervision and self.training:
            # Deep supervision outputs (need to upsample to match input size)
            ds1_out = F.interpolate(self.ds1(d1), size=x.shape[2:], mode='trilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(d2), size=x.shape[2:], mode='trilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(d3), size=x.shape[2:], mode='trilinear', align_corners=True)
            return logits, ds1_out, ds2_out, ds3_out
        
        return logits
    
    @staticmethod
    def get_memory_usage(input_shape=(1, 1, 128, 128, 64), base_filters=32, device='cuda'):
        """Estimate GPU memory usage for given input shape"""
        model = UNet3D(n_channels=1, n_classes=4, base_filters=base_filters, use_checkpoint=False)
        model = model.to(device)
        model.train()
        
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(input_shape).to(device)
        
        # Forward pass
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
    # Test configurations
    print("Testing 3D U-Net...")
    
    # Create model
    model = UNet3D(n_channels=1, n_classes=4, base_filters=32, trilinear=True, 
                   use_checkpoint=True, deep_supervision=False)
    
    # Test with dummy input
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
    model_ds = UNet3D(n_channels=1, n_classes=4, base_filters=32, deep_supervision=True)
    model_ds = model_ds.to(device)
    model_ds.train()
    
    outputs = model_ds(dummy_input)
    print(f"Main output shape: {outputs[0].shape}")
    print(f"DS outputs: {[o.shape for o in outputs[1:]]}")
