import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation (2D and 3D compatible)
    """
    def __init__(self, smooth=1.0, ignore_index=-100, include_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_background = include_background

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, C, H, W) or (B, C, D, H, W) - logits
            target: Ground truth (B, H, W) or (B, D, H, W) - class indices
        """
        # Ensure target is long type for one_hot encoding
        target = target.long()
        
        # Convert logits to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        n_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=n_classes)
        
        # Handle both 2D and 3D cases
        if pred.dim() == 4:  # 2D: (B, C, H, W)
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        else:  # 3D: (B, C, D, H, W)
            target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Calculate dice per class
        start_class = 0 if self.include_background else 1
        dice_scores = []
        
        for c in range(start_class, n_classes):
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = target_one_hot[:, c].contiguous().view(-1)
            
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_scores.append(dice)
        
        return 1 - torch.stack(dice_scores).mean()


class DiceLoss3D(nn.Module):
    """
    3D Dice Loss with per-class weighting option
    """
    def __init__(self, smooth=1.0, include_background=False, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.class_weights = class_weights

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - class indices
        """
        pred = torch.softmax(pred, dim=1)
        n_classes = pred.shape[1]
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), num_classes=n_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        start_idx = 0 if self.include_background else 1
        
        dice_scores = []
        weights = []
        
        for c in range(start_idx, n_classes):
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = target_one_hot[:, c].contiguous().view(-1)
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
            
            if self.class_weights is not None:
                weights.append(self.class_weights[c - start_idx])
            else:
                weights.append(1.0)
        
        dice_scores = torch.stack(dice_scores)
        weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
        weights = weights / weights.sum()
        
        return 1 - (dice_scores * weights).sum()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with alpha/beta parameters
    Better for highly imbalanced segmentation
    
    When alpha = beta = 0.5, equivalent to Dice loss
    alpha > beta penalizes false positives more
    alpha < beta penalizes false negatives more
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, include_background=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        n_classes = pred.shape[1]
        
        target_one_hot = F.one_hot(target.long(), num_classes=n_classes)
        if pred.dim() == 4:
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        else:
            target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        start_idx = 0 if self.include_background else 1
        tversky_scores = []
        
        for c in range(start_idx, n_classes):
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = target_one_hot[:, c].contiguous().view(-1)
            
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_scores.append(tversky)
        
        return 1 - torch.stack(tversky_scores).mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice Loss + Cross Entropy Loss (2D and 3D compatible)
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None, 
                 include_background=True):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(include_background=include_background)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred, target):
        # Ensure target is long type for both losses
        target = target.long()
        
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


class CombinedLoss3D(nn.Module):
    """
    Combined Dice + Cross Entropy loss optimized for 3D segmentation
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None,
                 include_background=False, focal_gamma=0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss3D(include_background=include_background)
        
        if focal_gamma > 0:
            self.ce_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, pred, target):
        target = target.long()
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


class DeepSupervisionLoss(nn.Module):
    """
    Loss wrapper for deep supervision
    Applies base loss to multiple outputs with decreasing weights
    """
    def __init__(self, base_loss: nn.Module, weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: List of outputs from different decoder levels, or single output
            target: Ground truth segmentation
        """
        if isinstance(outputs, (list, tuple)):
            total_loss = 0
            for i, (output, weight) in enumerate(zip(outputs, self.weights[:len(outputs)])):
                total_loss += weight * self.base_loss(output, target)
            return total_loss
        return self.base_loss(outputs, target)


# Test losses
if __name__ == "__main__":
    print("Testing 2D losses...")
    # 2D Dummy data
    batch_size, n_classes, h, w = 2, 4, 64, 64
    pred_2d = torch.randn(batch_size, n_classes, h, w)
    target_2d = torch.randint(0, n_classes, (batch_size, h, w))
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_value = dice_loss(pred_2d, target_2d)
    print(f"2D Dice Loss: {dice_value.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    combined_value = combined_loss(pred_2d, target_2d)
    print(f"2D Combined Loss: {combined_value.item():.4f}")
    
    print("\nTesting 3D losses...")
    # 3D Dummy data
    d = 32
    pred_3d = torch.randn(batch_size, n_classes, d, h, w)
    target_3d = torch.randint(0, n_classes, (batch_size, d, h, w))
    
    # Test 3D Dice Loss
    dice_loss_3d = DiceLoss3D()
    dice_value_3d = dice_loss_3d(pred_3d, target_3d)
    print(f"3D Dice Loss: {dice_value_3d.item():.4f}")
    
    # Test 3D Combined Loss
    combined_loss_3d = CombinedLoss3D(dice_weight=0.5, ce_weight=0.5)
    combined_value_3d = combined_loss_3d(pred_3d, target_3d)
    print(f"3D Combined Loss: {combined_value_3d.item():.4f}")
    
    # Test Tversky Loss
    tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
    tversky_value = tversky_loss(pred_3d, target_3d)
    print(f"Tversky Loss: {tversky_value.item():.4f}")
    
    # Test Deep Supervision Loss
    ds_outputs = [pred_3d, pred_3d, pred_3d]
    ds_loss = DeepSupervisionLoss(combined_loss_3d)
    ds_value = ds_loss(ds_outputs, target_3d)
    print(f"Deep Supervision Loss: {ds_value.item():.4f}")