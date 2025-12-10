import json
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset & Data Loading
# ==========================================

class PalmDataset(Dataset):
    """
    Standard dataset for Palm Images.
    Expects labels to be in a JSON file dictionary.
    """
    def __init__(self, image_paths, labels_dict, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels_dict = labels_dict
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        
        # 1. Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if failed (should be rare)
            image = Image.new('RGB', (224, 224))

        # 2. Get Label
        # Default to 0.5 if not found (or handle as error)
        scores = self.labels_dict.get(img_name, None)
        
        if scores is None:
            # If no label, return dummy for code stability but warn in real usage
            # For training, ideally we filtered these out already.
            label_tensor = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        else:
            label_tensor = torch.tensor([
                float(scores.get('strength', 0.5)),
                float(scores.get('romantic', 0.5)),
                float(scores.get('luck', 0.5)),
                float(scores.get('potential', 0.5))
            ], dtype=torch.float32)

        # 3. Apply Transforms
        if self.transform:
            image = self.transform(image)

        return image, label_tensor, str(img_path)

def load_labels(json_path):
    """
     Robust label loader. Handles nested stringified JSONs if present.
    """
    print(f"Loading labels from {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    clean_labels = {}
    
    # Check if list or dict
    if isinstance(raw_data, list):
        iterable = raw_data
    elif isinstance(raw_data, dict):
        iterable = raw_data.values() # assuming dict of entries
    else:
        print("Unknown JSON format")
        return {}

    count = 0
    for item in iterable:
        # Extract filename
        # Some formats might have full path, we just want basename
        img_ref = item.get('image', '')
        if not img_ref: continue
        
        img_name = os.path.basename(img_ref)
        
        # Extract scores
        scores_obj = item.get('scores', {})
        
        # Handle stringified scores
        if isinstance(scores_obj, str):
            try:
                scores_obj = json.loads(scores_obj)
            except:
                continue # Skip bad JSON
                
        # Validate keys
        if all(k in scores_obj for k in ['strength', 'romantic', 'luck', 'potential']):
            clean_labels[img_name] = scores_obj
            count += 1
            
    print(f"Successfully loaded {count} labeled entries.")
    return clean_labels

# ==========================================
# 2. Model Architecture
# ==========================================

class PalmResNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(PalmResNet, self).__init__()
        # Load Pretrained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze backbone if requested
        # Freeze backbone if requested (Freeze all except layer4)
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True # Ensure layer4 is trainable
        
        # Replace the final fully connected layer
        # ResNet18 fc input features is 512
        num_ftrs = self.backbone.fc.in_features
        
        # New Head: 512 -> 128 -> 4 (Sigmoid for 0-1 range)
        # Increased Dropout to 0.5 to prevent overfitting
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), # Regularization
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased from 0.3
            nn.Linear(128, 4),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 3. Training & Evaluation Functions
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def predict_single(model, image_path, transform, device):
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            return output.cpu().squeeze().numpy()
    except Exception as e:
        print(f"Failed prediction for {image_path}: {e}")
        return None

# ==========================================
# 4. Main Pipeline
# ==========================================

def main():
    # --- Config ---
    DATA_DIR = "./data/resized_dataset" # Adjust if needed
    LABEL_FILE = "./data/labels.json"
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {DEVICE}")

    # --- 1. Data Prep ---
    labels_dict = load_labels(LABEL_FILE)
    
    # Find images
    all_img_paths = []
    supported_exts = ['.jpg', '.jpeg', '.png']
    for ext in supported_exts:
        all_img_paths.extend(list(Path(DATA_DIR).glob(f"*{ext}")))
    
    # Filter only those with labels
    valid_paths = [str(p) for p in all_img_paths if os.path.basename(p) in labels_dict]
    
    if len(valid_paths) == 0:
        print("ERROR: No labeled images found! Check paths.")
        return

    print(f"Found {len(valid_paths)} images with labels.")

    # Split
    train_paths, val_paths = train_test_split(valid_paths, test_size=0.2, random_state=42)

    # Transforms (Standard ResNet ImageNet stats)
    # Using simple enhancement for training
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)), # Direct resize for val is often simpler than CenterCrop for palms
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = PalmDataset(train_paths, labels_dict, transform=train_tfm, is_training=True)
    val_ds = PalmDataset(val_paths, labels_dict, transform=val_tfm, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 2. Model Setup ---
    # --- 2. Model Setup ---
    # Freeze early layers, train layer4 + head
    model = PalmResNet(pretrained=True, freeze_backbone=True).to(DEVICE)
    
    # MSE Loss for regression
    criterion = nn.MSELoss()
    
    # Lower LR (5e-5), Moderate Weight Decay (1e-4)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=5e-5, weight_decay=1e-4)
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- 3. Training Loop ---
    best_loss = float('inf')
    history = {'train': [], 'val': []}

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        start_t = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        improved = ""
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            improved = "*"
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} {improved} "
              f"({time.time() - start_t:.1f}s)")

    print(f"\nTraining Complete. Best Val Loss: {best_loss:.4f}")
    
    # --- 4. Plot History ---
    plt.figure()
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('train_loss_curve.png')
    print("Saved learning curve to train_loss_curve.png")

    # --- 5. Sample Inference Check ---
    print("\nRunning Sample Inference Check on Validation Set...")
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()
    
    # Pick 3 random val images
    indices = np.random.choice(len(val_ds), min(3, len(val_ds)), replace=False)
    
    for i in indices:
        img, lbl, path = val_ds[i]
        # Need to undo transform for display ideally, but we just want to run forward pass
        # Unsqueeze for batch dim
        inp = img.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(inp).cpu().squeeze().numpy()
            
        print(f"\nImage: {os.path.basename(path)}")
        print(f"True: {lbl.numpy().round(2)}")
        print(f"Pred: {pred.round(2)}")

if __name__ == "__main__":
    main()
