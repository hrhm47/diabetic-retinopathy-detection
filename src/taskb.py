# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from sklearn.metrics import cohen_kappa_score
# from torch.utils.data import DataLoader
# import os

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths
# general_dataset_path = "/content/drive/MyDrive/diabetic_retinopathy_detection/data/raw/"  # Path to Kaggle DR Resized/APTOS dataset
# deepdrid_dataset_path = "./data/deepdrid_dataset"  # Path to DeepDRiD dataset
# model_save_path = "./models/fine_tuned_model.pth"

# # Define Transforms
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# # Load Datasets
# def load_datasets(data_path, transform):
#     return {
#         x: datasets.ImageFolder(os.path.join(data_path, x), transform[x])
#         for x in ['train', 'val']
#     }

# # Create DataLoaders
# def create_dataloaders(datasets, batch_size=32):
#     return {
#         x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
#         for x in ['train', 'val']
#     }

# # Load General Dataset
# general_datasets = load_datasets(general_dataset_path, data_transforms)
# general_dataloaders = create_dataloaders(general_datasets)

# # Load DeepDRiD Dataset
# deepdrid_datasets = load_datasets(deepdrid_dataset_path, data_transforms)
# deepdrid_dataloaders = create_dataloaders(deepdrid_datasets)

# # Define Model
# def initialize_model(model_name, num_classes, feature_extract=True):
#     if model_name == "resnet34":
#         model = models.resnet34(pretrained=True)
#         num_features = model.fc.in_features
#         model.fc = nn.Linear(num_features, num_classes)
#     else:
#         raise ValueError("Model not supported!")
#     return model.to(device)

# # Training Function
# def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print("-" * 30)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#             print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
#     return model

# # Stage 1: Fine-tuning on General Dataset
# model = initialize_model("resnet34", num_classes=len(general_datasets['train'].classes))
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# print("Stage 1: Training on General Dataset...")
# model = train_model(model, general_dataloaders, criterion, optimizer, num_epochs=10)

# # Save the model after Stage 1
# torch.save(model.state_dict(), model_save_path)

# # Stage 2: Fine-tuning on DeepDRiD
# print("Stage 2: Fine-tuning on DeepDRiD...")
# model = initialize_model("resnet34", num_classes=len(deepdrid_datasets['train'].classes))
# model.load_state_dict(torch.load(model_save_path))
# model = model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate for fine-tuning
# model = train_model(model, deepdrid_dataloaders, criterion, optimizer, num_epochs=5)

# # Save the final model
# torch.save(model.state_dict(), model_save_path)

# # Evaluation Function
# def evaluate_model(model, dataloader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     return cohen_kappa_score(all_labels, all_preds)

# # Evaluate Cohen Kappa Score
# print("Evaluating Cohen Kappa Score...")
# model.load_state_dict(torch.load(model_save_path))
# kappa_score = evaluate_model(model, deepdrid_dataloaders['val'])
# print(f"Cohen Kappa Score: {kappa_score:.4f}")


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Paths
cropped_dataset_path = "/content/drive/MyDrive/diabetic_retinopathy_detection/data/raw/resized_train_cropped"
cropped_labels_csv = "/content/drive/MyDrive/diabetic_retinopathy_detection/data/raw/trainLabel_cropped.csv"

resized_dataset_path = "/content/drive/MyDrive/diabetic_retinopathy_detection/data/raw/resized_train"
resized_labels_csv = "/content/drive/MyDrive/diabetic_retinopathy_detection/data/raw/trainLabel.csv"

# Select dataset
use_cropped = True  # Set False to use resized dataset
dataset_path = cropped_dataset_path if use_cropped else resized_dataset_path
labels_csv = cropped_labels_csv if use_cropped else resized_labels_csv

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Split dataset into train and val
full_labels = pd.read_csv(labels_csv)
train_labels, val_labels = train_test_split(full_labels, test_size=0.2, random_state=42)

train_labels.to_csv('train_split.csv', index=False)
val_labels.to_csv('val_split.csv', index=False)

# Create Datasets
datasets = {
    'train': DRDataset(csv_file='train_split.csv', img_dir=dataset_path, transform=data_transforms['train']),
    'val': DRDataset(csv_file='val_split.csv', img_dir=dataset_path, transform=data_transforms['val']),
}

# Create DataLoaders
dataloaders = {
    x: DataLoader(datasets[x], batch_size=32, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}

# Model selection and fine-tuning
model = models.resnet18(pretrained=True)  # Use ResNet18
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # 5 classes for DR
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training and validation loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

# Train and save the model
print("Starting training...")
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
torch.save(model.state_dict(), "fine_tuned_model.pth")
print("Model saved.")

# Fine-tune on DeepDRiD dataset
# Add similar logic here to load the DeepDRiD dataset and train further


