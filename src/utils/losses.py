import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    """
    def __init__(self, smooth=1.0, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, C, H, W) - logits
            target: Ground truth (B, H, W) - class indices
        """
        # Ensure target is long type for one_hot encoding
        target = target.long()
        
        # Convert logits to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        n_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)
        
        # Dice coefficient
        intersection = (pred * target_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target_one_hot.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice Loss + Cross Entropy Loss
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred, target):
        # Ensure target is long type for both losses
        target = target.long()
        
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


# Test losses
if __name__ == "__main__":
    # Dummy data
    batch_size, n_classes, h, w = 2, 4, 64, 64
    pred = torch.randn(batch_size, n_classes, h, w)  # Logits
    target = torch.randint(0, n_classes, (batch_size, h, w))  # Class indices
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_value = dice_loss(pred, target)
    print(f"Dice Loss: {dice_value.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    combined_value = combined_loss(pred, target)
    print(f"Combined Loss: {combined_value.item():.4f}")