"""
SegResNet: Semantic Segmentation Residual Network for 3D Medical Images

Architecture features:
- Encoder with residual blocks and downsampling
- Decoder with upsampling and skip connections
- Memory-efficient design with aggressive checkpointing
- VAE regularization option for better generalization

Reference: Myronenko, "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization", 
BrainLes 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Tuple


class GroupNorm3D(nn.GroupNorm):
    """Group Normalization for 3D data - more memory efficient than Instance Norm"""
    
    def __init__(self, num_channels, num_groups=8):
        # Ensure num_groups divides num_channels
        while num_channels % num_groups != 0:
            num_groups //= 2
        super().__init__(num_groups=num_groups, num_channels=num_channels, affine=True)


class ConvBNReLU(nn.Module):
    """Conv3D -> GroupNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.norm = GroupNorm3D(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection
    
    Structure: x -> Conv -> GN -> ReLU -> Conv -> GN -> + x -> ReLU
    """
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=kernel_size, 
                               padding=padding, bias=False)
        self.norm1 = GroupNorm3D(channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.norm2 = GroupNorm3D(channels)
        
        self.relu_out = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        return self.relu_out(out)


class DownsampleBlock(nn.Module):
    """Downsample block: strided convolution + residual blocks"""
    
    def __init__(self, in_channels, out_channels, n_blocks=1):
        super().__init__()
        
        # Strided convolution for downsampling
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            GroupNorm3D(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        blocks = [ResBlock(out_channels) for _ in range(n_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.downsample(x)
        return self.res_blocks(x)


class UpsampleBlock(nn.Module):
    """Upsample block: transposed conv or trilinear + conv + residual"""
    
    def __init__(self, in_channels, out_channels, n_blocks=1, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                GroupNorm3D(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                GroupNorm3D(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # 1x1 conv to combine skip connection
        self.combine = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            GroupNorm3D(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        blocks = [ResBlock(out_channels) for _ in range(n_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            diffZ = skip.size()[2] - x.size()[2]
            diffY = skip.size()[3] - x.size()[3]
            diffX = skip.size()[4] - x.size()[4]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.combine(x)
        return self.res_blocks(x)


class VAEDecoder(nn.Module):
    """
    Optional VAE regularization branch for better generalization
    Reconstructs downsampled input from latent space
    """
    
    def __init__(self, in_channels, latent_dim=128, output_shape=(128, 128, 64)):
        super().__init__()
        self.output_shape = output_shape
        
        # Encode to latent space
        self.fc_mu = nn.Linear(in_channels, latent_dim)
        self.fc_var = nn.Linear(in_channels, latent_dim)
        
        # Decode from latent space
        # Calculate intermediate sizes
        self.z_channels = 256
        d, h, w = output_shape[0] // 16, output_shape[1] // 16, output_shape[2] // 16
        self.init_size = (d, h, w)
        
        self.fc_decode = nn.Linear(latent_dim, self.z_channels * d * h * w)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.z_channels, 128, kernel_size=2, stride=2),
            GroupNorm3D(128), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            GroupNorm3D(64), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            GroupNorm3D(32), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            GroupNorm3D(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Global average pooling
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        
        # Encode
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x = self.fc_decode(z)
        x = x.view(-1, self.z_channels, *self.init_size)
        recon = self.decoder(x)
        
        return recon, mu, logvar


class SegResNet(nn.Module):
    """
    SegResNet: Encoder-Decoder with Residual Blocks for 3D Segmentation
    
    Features:
    - Efficient residual block design
    - Group normalization for stable training with small batches
    - Aggressive gradient checkpointing
    - Optional VAE regularization
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        init_filters: Initial number of filters (default 32)
        blocks_per_stage: Number of residual blocks per encoder stage
        trilinear: Use trilinear upsampling
        use_checkpoint: Enable gradient checkpointing
        use_vae: Enable VAE regularization branch
        deep_supervision: Enable deep supervision
    """
    
    def __init__(self, n_channels=1, n_classes=4, init_filters=32,
                 blocks_per_stage=(1, 2, 2, 4), trilinear=True,
                 use_checkpoint=True, use_vae=False, deep_supervision=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint
        self.use_vae = use_vae
        self.deep_supervision = deep_supervision
        
        f = init_filters
        
        # Initial convolution
        self.init_conv = ConvBNReLU(n_channels, f, kernel_size=3, stride=1, padding=1)
        
        # Initial residual blocks
        init_blocks = [ResBlock(f) for _ in range(blocks_per_stage[0])]
        self.init_res = nn.Sequential(*init_blocks)
        
        # Encoder
        self.down1 = DownsampleBlock(f, f*2, n_blocks=blocks_per_stage[1])      # 32->64
        self.down2 = DownsampleBlock(f*2, f*4, n_blocks=blocks_per_stage[2])    # 64->128
        self.down3 = DownsampleBlock(f*4, f*8, n_blocks=blocks_per_stage[3])    # 128->256
        
        # Bottleneck
        bottleneck_blocks = [ResBlock(f*8) for _ in range(blocks_per_stage[3])]
        self.bottleneck = nn.Sequential(*bottleneck_blocks)
        
        # Decoder
        self.up1 = UpsampleBlock(f*8, f*4, n_blocks=1, trilinear=trilinear)    # 256->128
        self.up2 = UpsampleBlock(f*4, f*2, n_blocks=1, trilinear=trilinear)    # 128->64
        self.up3 = UpsampleBlock(f*2, f, n_blocks=1, trilinear=trilinear)      # 64->32
        
        # Output head
        self.output_block = nn.Sequential(
            ResBlock(f),
            nn.Conv3d(f, n_classes, kernel_size=1)
        )
        
        # Deep supervision heads
        if deep_supervision:
            self.ds1 = nn.Conv3d(f*4, n_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(f*2, n_classes, kernel_size=1)
        
        # VAE branch
        if use_vae:
            self.vae = VAEDecoder(f*8, latent_dim=128)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _checkpointed_forward(self, module, x, *args):
        """Apply gradient checkpointing to a module"""
        if self.use_checkpoint and self.training:
            if args:
                return checkpoint(lambda inp, *a: module(inp, *a), x, *args, use_reentrant=False)
            return checkpoint(module, x, use_reentrant=False)
        return module(x, *args) if args else module(x)
    
    def forward(self, x):
        input_shape = x.shape[2:]
        
        # Initial convolution
        x0 = self.init_conv(x)
        x0 = self._checkpointed_forward(self.init_res, x0)
        
        # Encoder
        x1 = self._checkpointed_forward(self.down1, x0)
        x2 = self._checkpointed_forward(self.down2, x1)
        x3 = self._checkpointed_forward(self.down3, x2)
        
        # Bottleneck
        x3 = self._checkpointed_forward(self.bottleneck, x3)
        
        # VAE branch (only during training)
        vae_output = None
        if self.use_vae and self.training:
            vae_output = self.vae(x3)
        
        # Decoder
        d1 = self._checkpointed_forward(self.up1, x3, x2)
        d2 = self._checkpointed_forward(self.up2, d1, x1)
        d3 = self._checkpointed_forward(self.up3, d2, x0)
        
        # Output
        logits = self.output_block(d3)
        
        # Ensure output matches input size
        if logits.shape[2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode='trilinear', align_corners=True)
        
        # Return based on training mode and options
        if self.training:
            outputs = {'logits': logits}
            
            if self.deep_supervision:
                ds1_out = F.interpolate(self.ds1(d1), size=input_shape, mode='trilinear', align_corners=True)
                ds2_out = F.interpolate(self.ds2(d2), size=input_shape, mode='trilinear', align_corners=True)
                outputs['ds_outputs'] = [ds1_out, ds2_out]
            
            if self.use_vae and vae_output is not None:
                outputs['vae_recon'] = vae_output[0]
                outputs['vae_mu'] = vae_output[1]
                outputs['vae_logvar'] = vae_output[2]
            
            if self.deep_supervision or self.use_vae:
                return outputs
        
        return logits
    
    @staticmethod
    def get_memory_usage(input_shape=(1, 1, 128, 128, 64), init_filters=32, device='cuda'):
        """Estimate GPU memory usage"""
        model = SegResNet(n_channels=1, n_classes=4, init_filters=init_filters, use_checkpoint=False)
        model = model.to(device)
        model.train()
        
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(input_shape).to(device)
        
        y = model(x)
        if isinstance(y, dict):
            loss = y['logits'].sum()
        else:
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


class SegResNetLite(SegResNet):
    """
    Lighter version of SegResNet for memory-constrained scenarios
    Uses fewer filters and blocks
    """
    
    def __init__(self, n_channels=1, n_classes=4, **kwargs):
        # Reduce default parameters
        kwargs.setdefault('init_filters', 16)
        kwargs.setdefault('blocks_per_stage', (1, 1, 1, 2))
        super().__init__(n_channels, n_classes, **kwargs)


# Test the model
if __name__ == "__main__":
    print("Testing SegResNet...")
    
    # Create model
    model = SegResNet(n_channels=1, n_classes=4, init_filters=32, 
                      use_checkpoint=True, deep_supervision=False, use_vae=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Test input shape: (batch, channels, depth, height, width)
    dummy_input = torch.randn(2, 1, 64, 128, 128).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.train()
    output = model(dummy_input)
    if isinstance(output, dict):
        print(f"Output shape: {output['logits'].shape}")
    else:
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
    
    # Test with VAE and deep supervision
    print("\n--- Testing with VAE + Deep Supervision ---")
    model_full = SegResNet(n_channels=1, n_classes=4, init_filters=32,
                           use_vae=True, deep_supervision=True)
    model_full = model_full.to(device)
    model_full.train()
    
    outputs = model_full(dummy_input)
    print(f"Main output shape: {outputs['logits'].shape}")
    print(f"DS outputs: {[o.shape for o in outputs['ds_outputs']]}")
    print(f"VAE recon shape: {outputs['vae_recon'].shape}")
    
    # Test SegResNetLite
    print("\n--- Testing SegResNetLite ---")
    model_lite = SegResNetLite(n_channels=1, n_classes=4)
    model_lite = model_lite.to(device)
    
    output_lite = model_lite(dummy_input)
    if isinstance(output_lite, dict):
        print(f"Lite output shape: {output_lite['logits'].shape}")
    else:
        print(f"Lite output shape: {output_lite.shape}")
    
    lite_params = sum(p.numel() for p in model_lite.parameters())
    print(f"Lite parameters: {lite_params:,}")
