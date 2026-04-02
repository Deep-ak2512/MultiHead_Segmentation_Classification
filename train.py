import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from custom_dataset import create_segmentation_dataloaders
from network import UNet_Segmentation
from loss import DiceLoss, IoULoss, CombinedLoss, calculate_iou, calculate_dice, calculate_accuracy


def train_model(resume=False):
        
    # Configuration
    DATA_DIR = "../data/Dataset_BUSI_with_GT"
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    IMG_SIZE = 256
    CHECKPOINT_DIR = "./checkpoints"
    
    # Checkpoint path
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_segmentation_model.pth')
    history_path = os.path.join(CHECKPOINT_DIR, 'segmentation_training_history.pth')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_segmentation_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
        train_split=0.8
    )
    
    # Initialize custom segmentation model for 3-class segmentation
    print("\nInitializing custom UNet segmentation model for 3-class segmentation...")
    model = UNet_Segmentation(in_channels=1, n_classes=3)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function: Combined Dice + IoU Loss
    criterion = CombinedLoss(alpha=0.5, smooth=1.0)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    
    # Training variables
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_iou = 0
    best_val_dice = 0
    best_val_seg_accuracy = 0
    best_val_cls_accuracy = 0
    best_epoch = 0
    train_loss_history = []
    val_loss_history = []
    val_iou_history = []
    val_dice_history = []
    val_seg_accuracy_history = []
    val_cls_accuracy_history = []
    
    # Resume from checkpoint if requested
    if resume and os.path.exists(checkpoint_path):
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        best_val_iou = checkpoint.get('val_iou', 0)
        best_val_dice = checkpoint.get('val_dice', 0)
        best_epoch = start_epoch
        
        # Load training history if available
        if 'train_loss_history' in checkpoint:
            train_loss_history = checkpoint['train_loss_history']
            val_loss_history = checkpoint['val_loss_history']
            val_iou_history = checkpoint.get('val_iou_history', [])
            val_dice_history = checkpoint.get('val_dice_history', [])
            print(f"Loaded training history up to epoch {start_epoch}")
        elif os.path.exists(history_path):
            history = torch.load(history_path)
            train_loss_history = history['train_loss_history']
            val_loss_history = history['val_loss_history']
            val_iou_history = history.get('val_iou_history', [])
            val_dice_history = history.get('val_dice_history', [])
            print(f"Loaded training history from: {history_path}")
        
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
        print(f"Best validation IoU so far: {best_val_iou:.4f}")
        print(f"Best validation Dice so far: {best_val_dice:.4f}")
        print(f"{'='*60}\n")
    else:
        if resume:
            print(f"\nWarning: Resume requested but no checkpoint found at {checkpoint_path}")
            print(f"Starting training from scratch...\n")
        print(f"Model initialized with custom UNet architecture for 3-class segmentation")
        print(f"Classes: Normal, Malignant, Benign")
        print(f"Using Combined Loss (Dice + IoU) for multi-class")
    
    print(f"\n{'='*60}")
    print(f"Starting Training from epoch {start_epoch + 1} to {NUM_EPOCHS}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # ===================== Training Phase =====================
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Forward pass
            seg_outputs, cls_outputs = model(images)
            seg_loss = criterion(seg_outputs, masks)
            cls_loss = nn.CrossEntropyLoss()(cls_outputs, labels)
            loss = seg_loss + 0.5 * cls_loss  # Weight classification loss less
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        
        # Print training loss
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {epoch_train_loss:.4f}")
        
        # Update learning rate scheduler
        scheduler.step()
        
        # ===================== Validation Phase =====================
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_dice_sum = 0.0
        val_seg_accuracy_sum = 0.0
        val_cls_accuracy_sum = 0.0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                seg_outputs, cls_outputs = model(images)
                seg_loss = criterion(seg_outputs, masks)
                cls_loss = nn.CrossEntropyLoss()(cls_outputs, labels)
                loss = seg_loss + 0.5 * cls_loss
                val_loss += loss.item()
                
                # Calculate segmentation metrics
                val_iou_sum += calculate_iou(seg_outputs, masks)
                val_dice_sum += calculate_dice(seg_outputs, masks)
                val_seg_accuracy_sum += calculate_accuracy(seg_outputs, masks)
                
                # Calculate classification accuracy
                cls_preds = torch.argmax(cls_outputs, dim=1)
                cls_accuracy = (cls_preds == labels).float().mean()
                val_cls_accuracy_sum += cls_accuracy.item()
        
        val_loss = val_loss / len(val_loader)
        val_iou = val_iou_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        val_seg_accuracy = val_seg_accuracy_sum / len(val_loader)
        val_cls_accuracy = val_cls_accuracy_sum / len(val_loader)
        
        # Save history
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(val_loss)
        val_iou_history.append(val_iou)
        val_dice_history.append(val_dice)
        val_seg_accuracy_history.append(val_seg_accuracy)
        val_cls_accuracy_history.append(val_cls_accuracy)
        
        # Print validation results
        print(f"  - Validation Loss: {val_loss:.4f} - Seg Acc: {val_seg_accuracy*100:.2f}% - Cls Acc: {val_cls_accuracy*100:.2f}% - IoU: {val_iou:.4f} - Dice: {val_dice:.4f}")
        
        # ===================== Check for Best Model =====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou = val_iou
            best_val_dice = val_dice
            best_val_seg_accuracy = val_seg_accuracy
            best_val_cls_accuracy = val_cls_accuracy
            best_epoch = epoch + 1
            
            print(f"\n{'='*60}")
            print(f"*** NEW BEST VALIDATION LOSS: {best_val_loss:.4f} at Epoch {best_epoch} ***")
            print(f"Seg Accuracy: {best_val_seg_accuracy*100:.2f}% - Cls Accuracy: {best_val_cls_accuracy*100:.2f}% - IoU: {best_val_iou:.4f} - Dice: {best_val_dice:.4f}")
            print(f"{'='*60}")
            
            # Save model checkpoint
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_iou': best_val_iou,
                'val_dice': best_val_dice,
                'val_seg_accuracy': best_val_seg_accuracy,
                'val_cls_accuracy': best_val_cls_accuracy,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'val_iou_history': val_iou_history,
                'val_dice_history': val_dice_history,
                'val_seg_accuracy_history': val_seg_accuracy_history,
                'val_cls_accuracy_history': val_cls_accuracy_history,
            }, checkpoint_path)
            
            print(f"Model saved to: {checkpoint_path}\n")
    
    # ===================== Training Complete =====================
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Best Segmentation Accuracy: {best_val_seg_accuracy*100:.2f}% at Epoch {best_epoch}")
    print(f"Best Classification Accuracy: {best_val_cls_accuracy*100:.2f}% at Epoch {best_epoch}")
    print(f"Best Validation IoU: {best_val_iou:.4f} at Epoch {best_epoch}")
    print(f"Best Validation Dice: {best_val_dice:.4f} at Epoch {best_epoch}")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Save training history for visualization
    torch.save({
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'val_iou_history': val_iou_history,
        'val_dice_history': val_dice_history,
        'val_seg_accuracy_history': val_seg_accuracy_history,
        'val_cls_accuracy_history': val_cls_accuracy_history,
        'best_val_loss': best_val_loss,
        'best_val_iou': best_val_iou,
        'best_val_dice': best_val_dice,
        'best_val_seg_accuracy': best_val_seg_accuracy,
        'best_val_cls_accuracy': best_val_cls_accuracy,
        'best_epoch': best_epoch,
    }, history_path)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    import sys
    resume = '--resume' in sys.argv
    train_model(resume=resume)
