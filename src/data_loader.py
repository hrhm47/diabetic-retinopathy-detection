#       a) Fine-tune a pretrained model using the DeepDRiD dataset. (5 points) 




import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ========== preparing the dataset ============


# Define paths
data_dir = "./data/raw/DeepDRiD/"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(15),
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean/std
])

val_test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check dataset details
print("\n\n       dataset details ")
print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"\nValidation dataset size: {len(val_dataset)}")
print(f"\nTest dataset size: {len(test_dataset)}\n")




#  ==== Loading and Fine-Tuning a Pretrained Model ===========
