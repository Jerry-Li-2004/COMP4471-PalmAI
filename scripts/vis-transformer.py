import warnings
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

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
            label_tensor = torch.tensor(
                [0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
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
        iterable = raw_data.values()  # assuming dict of entries
    else:
        print("Unknown JSON format")
        return {}

    count = 0
    for item in iterable:
        # Extract filename
        # Some formats might have full path, we just want basename
        img_ref = item.get('image', '')
        if not img_ref:
            continue

        img_name = os.path.basename(img_ref)

        # Extract scores
        scores_obj = item.get('scores', {})

        # Handle stringified scores
        if isinstance(scores_obj, str):
            try:
                scores_obj = json.loads(scores_obj)
            except:
                continue  # Skip bad JSON

        # Validate keys
        if all(k in scores_obj for k in ['strength', 'romantic', 'luck', 'potential']):
            clean_labels[img_name] = scores_obj
            count += 1

    print(f"Successfully loaded {count} labeled entries.")
    return clean_labels

# ==========================================
# 2. Transformer-based Model Architecture
# ==========================================


class PalmVisionTransformer(nn.Module):
    def __init__(self,
                 model_name='vit_base_patch16_224',
                 pretrained=True,
                 freeze_backbone=False,
                 dropout_rate=0.5,
                 hidden_dim=512,
                 num_regression_heads=4):
        """
        Vision Transformer for palm image regression.

        Args:
            model_name: Which ViT variant to use
            pretrained: Use pretrained weights
            freeze_backbone: Freeze transformer backbone
            dropout_rate: Dropout rate for regularization
            hidden_dim: Hidden dimension for regression head
            num_regression_heads: Number of output scores (4 for palm reading)
        """
        super(PalmVisionTransformer, self).__init__()

        # Import transformer models
        try:
            from transformers import ViTModel, ViTConfig, ViTForImageClassification
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers")

        # Load pretrained ViT model
        if pretrained:
            if model_name == 'vit_base_patch16_224':
                print("Loading ViT-Base pretrained weights...")
                self.vit = ViTModel.from_pretrained(
                    'google/vit-base-patch16-224-in21k')
            elif model_name == 'vit_base_patch16_384':
                print("Loading ViT-Base-384 pretrained weights...")
                self.vit = ViTModel.from_pretrained(
                    'google/vit-base-patch16-384')
            elif model_name == 'vit_small_patch16_224':
                print("Loading ViT-Small pretrained weights...")
                self.vit = ViTModel.from_pretrained(
                    'WinKawaks/vit-small-patch16-224')
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        else:
            # Initialize from scratch
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072
            )
            self.vit = ViTModel(config)

        # Get hidden dimension
        hidden_size = self.vit.config.hidden_size

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.vit.named_parameters():
                if "encoder.layer.11" not in name:  # Only unfreeze last layer
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_regression_heads),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        # Initialize regression head
        self._init_weights()

    def _init_weights(self):
        """Initialize regression head weights"""
        for module in self.regression_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through ViT

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            scores: Tensor of shape (batch_size, 4) with regression scores
        """
        # Get ViT features
        outputs = self.vit(x)

        # Use CLS token for classification/regression
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Pass through regression head
        scores = self.regression_head(cls_token)

        return scores


class PalmViTWithCNN(nn.Module):
    """
    Hybrid model: CNN feature extractor + Transformer encoder
    """

    def __init__(self, cnn_backbone='resnet18', pretrained=True,
                 num_regression_heads=4, hidden_dim=512, dropout_rate=0.5):
        super(PalmViTWithCNN, self).__init__()

        # 1. CNN Backbone
        if cnn_backbone == 'resnet18':
            cnn = models.resnet18(
                weights='IMAGENET1K_V1' if pretrained else None)
            cnn_features = 512
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        elif cnn_backbone == 'efficientnet_b0':
            cnn = models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None)
            cnn_features = 1280
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_features,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # 3. Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cnn_features))

        # 4. Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(cnn_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_regression_heads),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn(x)  # Shape: (batch_size, features, H, W)

        # Reshape for transformer: (batch_size, features, H*W) -> (batch_size, H*W, features)
        batch_size, features, h, w = cnn_features.shape
        cnn_features = cnn_features.view(
            batch_size, features, -1).permute(0, 2, 1)

        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((cls_tokens, cnn_features), dim=1)

        # Transformer encoding
        encoded = self.transformer_encoder(transformer_input)

        # Get CLS token output
        cls_output = encoded[:, 0, :]

        # Regression
        scores = self.regression_head(cls_output)

        return scores

# ==========================================
# 3. Training & Evaluation Functions
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip_value=None, grad_accum_steps=1):
    """
    Train for one epoch with gradient accumulation
    """
    model.train()
    running_loss = 0.0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_value)

            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * images.size(0) * grad_accum_steps
        pbar.set_postfix({'loss': loss.item() * grad_accum_steps})

    # Handle remaining gradients
    if (batch_idx + 1) % grad_accum_steps != 0:
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

    return epoch_loss, mae, rmse


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
# 4. Main Training Pipeline
# ==========================================


def train_transformer_model():
    # --- Config ---
    DATA_DIR = "./data/resized_dataset"  # Adjust if needed
    LABEL_FILE = "./data/labels.json"
    BATCH_SIZE = 8  # Smaller batch size for transformer
    LR = 1e-4
    EPOCHS = 30
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {DEVICE}")

    # Model config
    MODEL_TYPE = 'vit_base'  # Options: 'vit_base', 'vit_small', 'hybrid'
    PRETRAINED = True
    FREEZE_BACKBONE = False
    GRADIENT_ACCUMULATION_STEPS = 2
    GRADIENT_CLIP = 1.0

    # --- 1. Data Prep ---
    labels_dict = load_labels(LABEL_FILE)

    # Find images
    all_img_paths = []
    supported_exts = ['.jpg', '.jpeg', '.png']
    for ext in supported_exts:
        all_img_paths.extend(list(Path(DATA_DIR).glob(f"*{ext}")))

    # Filter only those with labels
    valid_paths = [str(p)
                   for p in all_img_paths if os.path.basename(p) in labels_dict]

    if len(valid_paths) == 0:
        print("ERROR: No labeled images found! Check paths.")
        return

    print(f"Found {len(valid_paths)} images with labels.")

    # Split
    train_paths, val_paths = train_test_split(
        valid_paths, test_size=0.2, random_state=42)

    # Transforms for Vision Transformer
    # Standard ViT transforms (similar to ImageNet but with 224x224)
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # ViT typically uses 0.5 mean/std
                             std=[0.5, 0.5, 0.5])
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_ds = PalmDataset(train_paths, labels_dict,
                           transform=train_tfm, is_training=True)
    val_ds = PalmDataset(val_paths, labels_dict,
                         transform=val_tfm, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    # --- 2. Model Setup ---
    if MODEL_TYPE == 'vit_base':
        model = PalmVisionTransformer(
            model_name='vit_base_patch16_224',
            pretrained=PRETRAINED,
            freeze_backbone=FREEZE_BACKBONE,
            dropout_rate=0.3,
            hidden_dim=512,
            num_regression_heads=4
        )
    elif MODEL_TYPE == 'vit_small':
        model = PalmVisionTransformer(
            model_name='vit_small_patch16_224',
            pretrained=PRETRAINED,
            freeze_backbone=FREEZE_BACKBONE,
            dropout_rate=0.3,
            hidden_dim=256,
            num_regression_heads=4
        )
    elif MODEL_TYPE == 'hybrid':
        model = PalmViTWithCNN(
            cnn_backbone='resnet18',
            pretrained=PRETRAINED,
            num_regression_heads=4,
            hidden_dim=256,
            dropout_rate=0.3
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {MODEL_TYPE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()  # Alternative: Huber loss

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,  # Weight decay for transformers
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader) // GRADIENT_ACCUMULATION_STEPS + 1,
        pct_start=0.1  # 10% warmup
    )

    # --- 3. Training Loop ---
    best_val_loss = float('inf')
    best_mae = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }

    print(f"\nStarting {MODEL_TYPE.upper()} Training for {EPOCHS} epochs...")
    print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")

    for epoch in range(EPOCHS):
        start_t = time.time()

        # Training
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            grad_clip_value=GRADIENT_CLIP,
            grad_accum_steps=GRADIENT_ACCUMULATION_STEPS
        )

        # Validation
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion, DEVICE)

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # Save best model
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae = val_mae
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, f'best_{MODEL_TYPE}_model.pth')
            improved = "*"

        # Print epoch stats
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"MAE: {val_mae:.4f} | "
              f"RMSE: {val_rmse:.4f} | "
              f"LR: {current_lr:.2e} {improved} "
              f"({time.time() - start_t:.1f}s)")

    print(f"\nTraining Complete.")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best MAE: {best_mae:.4f}")

    # --- 4. Plot Training Curves ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history['val_mae'], label='MAE', color='orange')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history['val_rmse'], label='RMSE', color='green')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{MODEL_TYPE}_training_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {MODEL_TYPE}_training_curves.png")

    # --- 5. Sample Inference Check ---
    print(f"\nRunning Sample Inference Check on Validation Set...")

    # Load best model
    checkpoint = torch.load(
        f'best_{MODEL_TYPE}_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Pick 3 random val images
    indices = np.random.choice(len(val_ds), min(5, len(val_ds)), replace=False)

    print("\n" + "="*60)
    print(f"{'Image':<30} {'True':<30} {'Predicted':<30}")
    print("-"*60)

    for i in indices:
        img, lbl, path = val_ds[i]
        # Need to undo transform for display ideally, but we just want to run forward pass
        inp = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(inp).cpu().squeeze().numpy()

        true_vals = lbl.numpy()
        pred_vals = pred

        # Calculate absolute error
        abs_error = np.abs(true_vals - pred_vals)

        print(f"{os.path.basename(path):<30}")
        print(
            f"{'':<30} Strength:  {true_vals[0]:.3f} -> {pred_vals[0]:.3f} (err: {abs_error[0]:.3f})")
        print(
            f"{'':<30} Romantic:  {true_vals[1]:.3f} -> {pred_vals[1]:.3f} (err: {abs_error[1]:.3f})")
        print(
            f"{'':<30} Luck:      {true_vals[2]:.3f} -> {pred_vals[2]:.3f} (err: {abs_error[2]:.3f})")
        print(
            f"{'':<30} Potential: {true_vals[3]:.3f} -> {pred_vals[3]:.3f} (err: {abs_error[3]:.3f})")
        print("-"*60)

    return model, history

# ==========================================
# 5. Inference Function
# ==========================================


def load_trained_transformer_model(model_path, model_type='vit_base', device='cuda'):
    """
    Load trained transformer model for inference
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if model_type == 'vit_base':
        model = PalmVisionTransformer(
            model_name='vit_base_patch16_224',
            pretrained=False,
            freeze_backbone=False,
            dropout_rate=0.0,  # No dropout during inference
            hidden_dim=512,
            num_regression_heads=4
        )
    elif model_type == 'vit_small':
        model = PalmVisionTransformer(
            model_name='vit_small_patch16_224',
            pretrained=False,
            freeze_backbone=False,
            dropout_rate=0.0,
            hidden_dim=256,
            num_regression_heads=4
        )
    elif model_type == 'hybrid':
        model = PalmViTWithCNN(
            cnn_backbone='resnet18',
            pretrained=False,
            num_regression_heads=4,
            hidden_dim=256,
            dropout_rate=0.0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Best validation MAE: {checkpoint['val_mae']:.4f}")

    return model


def predict_palm_image(image_path, model, device='cuda'):
    """
    Predict palm scores for a single image
    """
    # Transform for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return predict_single(model, image_path, transform, device)

# ==========================================
# 6. Main Execution
# ==========================================


if __name__ == "__main__":
    # Install required packages if not installed
    try:
        from transformers import ViTModel, ViTConfig
    except ImportError:
        print("Installing transformers package...")
        import subprocess
        subprocess.check_call(["pip", "install", "transformers"])
        from transformers import ViTModel, ViTConfig

    # Train the transformer model
    trained_model, training_history = train_transformer_model()

    # Optional: Compare with different model types
    print("\n" + "="*60)
    print("To train with different model types, modify MODEL_TYPE variable:")
    print("Options: 'vit_base', 'vit_small', or 'hybrid'")
    print("="*60)

    # Example inference after training
    print("\nExample: Load trained model and make predictions")
    print("="*60)

    # This assumes you want to use the vit_base model
    MODEL_TYPE = 'vit_base'  # Change as needed

    try:
        loaded_model = load_trained_transformer_model(
            model_path=f'best_{MODEL_TYPE}_model.pth',
            model_type=MODEL_TYPE,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Test on a sample image (use first validation image)
        DATA_DIR = "./data/resized_dataset"
        all_img_paths = list(Path(DATA_DIR).glob("*.jpg")) + \
            list(Path(DATA_DIR).glob("*.png"))
        if all_img_paths:
            sample_image = str(all_img_paths[0])
            print(f"\nMaking prediction on: {sample_image}")
            prediction = predict_palm_image(sample_image, loaded_model)

            if prediction is not None:
                print("\nPredicted Scores:")
                print(f"  Strength:  {prediction[0]:.3f}")
                print(f"  Romantic:  {prediction[1]:.3f}")
                print(f"  Luck:      {prediction[2]:.3f}")
                print(f"  Potential: {prediction[3]:.3f}")
    except Exception as e:
        print(f"Error during inference: {e}")
