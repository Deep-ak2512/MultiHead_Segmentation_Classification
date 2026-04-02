import os
from PIL import Image
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class BUSISegmentationDataset(Dataset):
    
    def __init__(self, image_mask_pairs, indices=None, transform=None):

        self.image_mask_pairs = image_mask_pairs
        self.indices = indices if indices is not None else range(len(image_mask_pairs))
        self.class_id = {"normal":0,"benign":1,"malignant":2}
        
        self.transform = transform
        
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get actual index from train/val indices
        actual_idx = self.indices[idx]
        image_path, mask_path = self.image_mask_pairs[actual_idx]
        
        # Load image as grayscale (1 channel)
        image = Image.open(image_path).convert('L')
        
        # Load mask and convert to binary
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)  # Convert PIL Image to NumPy array
        mask = (mask > 127).astype(np.float32)
        mask = Image.fromarray((mask * 255).astype(np.uint8))  # Convert back to PIL Image
        
        if "benign" in image_path:
            label = self.class_id["benign"]
        elif "malignant" in image_path:
            label = self.class_id["malignant"]
        else:
            label = self.class_id["normal"]
        
        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask, label


class Compose:
    """
    Custom compose class to apply transforms to both image and mask
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ToTensor:
    """Convert PIL images to PyTorch tensors"""
    def __call__(self, image, mask):
        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)
        return image, mask


class Resize:
    """Resize both image and mask to target size"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, mask):
        image = transforms.functional.resize(image, self.size)
        mask = transforms.functional.resize(mask, self.size)
        return image, mask


class RandomHorizontalFlip:
    """Randomly flip both image and mask horizontally"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    """Randomly flip both image and mask vertically"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return image, mask


class RandomRotation:
    """Randomly rotate both image and mask"""
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return image, mask


class Normalize:
    """Normalize image (mask is not normalized)"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, mask):
        image = transforms.functional.normalize(image, self.mean, self.std)
        return image, mask


def get_segmentation_transforms(img_size=256):
  
    # Training transforms with data augmentation (grayscale images)
    train_transform = Compose([
        Resize((img_size, img_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])
    
    # Validation transforms (no augmentation)
    val_transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])
    
    return train_transform, val_transform


def load_busi_dataset(data_dir):

    categories = ['benign', 'malignant', 'normal']
    all_pairs = []
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: {category} folder not found at {category_path}")
            continue
        
        # Get all mask files
        mask_pattern = os.path.join(category_path, '*_mask.png')
        mask_files = glob.glob(mask_pattern)
        
        for mask_path in mask_files:
            # Get corresponding image path
            image_path = mask_path.replace('_mask.png', '.png')
            
            if os.path.exists(image_path):
                all_pairs.append((image_path, mask_path))
            else:
                print(f"Warning: Image not found for mask {mask_path}")
        
        print(f"Loaded {len([p for p in all_pairs if category in p[0]])} pairs from {category}")
    
    print(f"\nTotal image-mask pairs: {len(all_pairs)}")
    return all_pairs


def create_segmentation_dataloaders(data_dir, batch_size=8, num_workers=4, img_size=256, train_split=0.8):

    # Load all image-mask pairs
    image_mask_pairs = load_busi_dataset(data_dir)
    
    if len(image_mask_pairs) == 0:
        raise ValueError("No image-mask pairs found in the dataset!")
    
    # Create train/val split
    n_total = len(image_mask_pairs)
    n_train = int(n_total * train_split)
    
    # Shuffle indices
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    
    # Get transforms
    train_transform, val_transform = get_segmentation_transforms(img_size)
    
    # Create datasets
    train_dataset = BUSISegmentationDataset(
        image_mask_pairs=image_mask_pairs,
        indices=train_indices,
        transform=train_transform
    )
    
    val_dataset = BUSISegmentationDataset(
        image_mask_pairs=image_mask_pairs,
        indices=val_indices,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test the dataset and dataloader
    data_dir = "../data/Dataset_BUSI_with_GT"
    
    print("="*60)
    print("Testing BUSI Segmentation Dataloader")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader = create_segmentation_dataloaders(
        data_dir=data_dir,
        batch_size=1,
        num_workers=2,
        img_size=256,
        train_split=0.8
    )
    
    print(f"\n{'='*60}")
    print("Testing dataloader...")
    print(f"{'='*60}")
    
    # Test loading a batch from training loader
    for batch_idx, (images, masks,labels) in enumerate(train_loader):
        print(f"\nTraining Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Masks dtype: {masks.dtype}")
        
        print(f"  Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
        print(f"  Value: {labels}")
        
        if batch_idx == 0:  # Only test first batch
            break
    
    # Test loading a batch from validation loader
    for batch_idx, (images, masks,labels) in enumerate(val_loader):
        print(f"\nValidation Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  label  shape: {labels.shape}")
        
        if batch_idx == 0:  # Only test first batch
            break
    
    print(f"\n{'='*60}")
    print("Dataset and dataloader working correctly!")
    print(f"{'='*60}")
