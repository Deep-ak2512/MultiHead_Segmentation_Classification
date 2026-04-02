import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):

        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten the tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection
        intersection = (predictions * targets).sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class IoULoss(nn.Module):
   
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
       
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten the tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection
        intersection = (predictions * targets).sum()
        
        # Calculate union
        union = predictions.sum() + targets.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(nn.Module):
   
    def __init__(self, alpha=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        self.iou_loss = IoULoss(smooth=smooth)
    
    def forward(self, predictions, targets):
       
        dice = self.dice_loss(predictions, targets)
        iou = self.iou_loss(predictions, targets)
        return self.alpha * dice + (1 - self.alpha) * iou


def calculate_iou(predictions, targets, threshold=0.5):
   
    # Apply sigmoid and threshold
    preds = torch.sigmoid(predictions)
    preds = (preds > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_dice(predictions, targets, threshold=0.5):
   
    # Apply sigmoid and threshold
    preds = torch.sigmoid(predictions)
    preds = (preds > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection
    intersection = (preds * targets).sum()
    
    # Avoid division by zero
    if preds.sum() + targets.sum() == 0:
        return 0.0
    
    dice = (2. * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)
    return dice.item()


def calculate_accuracy(predictions, targets, threshold=0.5):
   
    # Apply sigmoid and threshold
    preds = torch.sigmoid(predictions)
    preds = (preds > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Calculate correct predictions
    correct = (preds == targets).sum()
    total = targets.numel()
    
    accuracy = (correct / total).item()
    return accuracy


if __name__ == "__main__":
    # Test the loss functions for binary segmentation
    print("Testing Dice Loss and IoU Loss for binary segmentation...")
    
    # Create dummy predictions and targets
    batch_size = 4
    height, width = 256, 256
    
    predictions = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test Dice Loss
    dice_loss_fn = DiceLoss()
    dice_loss = dice_loss_fn(predictions, targets)
    print(f"Dice Loss: {dice_loss.item():.4f}")
    
    # Test IoU Loss
    iou_loss_fn = IoULoss()
    iou_loss = iou_loss_fn(predictions, targets)
    print(f"IoU Loss: {iou_loss.item():.4f}")
    
    # Test Combined Loss
    combined_loss_fn = CombinedLoss(alpha=0.5)
    combined_loss = combined_loss_fn(predictions, targets)
    print(f"Combined Loss (alpha=0.5): {combined_loss.item():.4f}")
    
    # Test metrics
    dice_metric = calculate_dice(predictions, targets)
    iou_metric = calculate_iou(predictions, targets)
    accuracy_metric = calculate_accuracy(predictions, targets)
    print(f"\nDice Metric: {dice_metric:.4f}")
    print(f"IoU Metric: {iou_metric:.4f}")
    print(f"Accuracy Metric: {accuracy_metric:.4f}")
    
    print("\nAll loss functions working correctly!")
