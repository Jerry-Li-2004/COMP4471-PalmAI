import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from torchvision import models
import timm


class PalmLineProcessor:
    """Pre-processes palm images to enhance line detection"""
    
    def __init__(self):
        self.line_enhancement_kernel = np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    
    def preprocess_image(self, image):
        """Enhance palm lines for better feature extraction"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Line enhancement
        enhanced = cv2.filter2D(enhanced, -1, self.line_enhancement_kernel)
        
        # Noise reduction
        enhanced = cv2.medianBlur(enhanced, 3)
        
        return Image.fromarray(enhanced)


class PalmScoreDataset(Dataset):
    """Dataset for palm score regression with real labels"""
    
    def __init__(self, image_paths, labels_dict=None, transform=None, is_training=True):
        self.image_paths = image_paths
        self.transform = transform
        self.is_training = is_training
        self.processor = PalmLineProcessor()
        self.labels_dict = labels_dict or {}
        
        # Filter images that have labels
        self.valid_indices = []
        for idx, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            if img_name in self.labels_dict:
                self.valid_indices.append(idx)
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        image_path = self.image_paths[actual_idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess to enhance palm lines
            processed_image = self.processor.preprocess_image(image)
            
            if self.transform:
                processed_image = self.transform(processed_image)
            
            if self.is_training and self.labels_dict:
                img_name = os.path.basename(image_path)
                scores = self.labels_dict.get(img_name, {
                    'strength': 0.5, 'romantic': 0.5, 'luck': 0.5, 'potential': 0.5
                })
                
                # Convert to tensor
                label = torch.tensor([
                    scores.get('strength', 0.5),
                    scores.get('romantic', 0.5),
                    scores.get('luck', 0.5),
                    scores.get('potential', 0.5)
                ], dtype=torch.float32)
                
                return processed_image, label
            
            return processed_image
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            dummy_image = torch.zeros(1, 224, 224) if self.transform else Image.new('L', (224, 224))
            if self.is_training:
                dummy_label = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
                return dummy_image, dummy_label
            return dummy_image


class MultiScaleCNNPredictor(nn.Module):
    """Multi-scale CNN for palm score regression"""
    
    def __init__(self, in_channels=1):
        super(MultiScaleCNNPredictor, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global feature extraction
        self.global_features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Regression head for 4 scores
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # 4 output scores
        )
        
        # Sigmoid activation for scores between 0-1
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Multi-scale feature extraction
        x1 = self.scale1(x)
        x2 = self.scale2(x1)
        x3 = self.scale3(x2)
        
        # Attention on mid-level features
        attn = self.attention(x2)
        attended_features = x2 * attn
        
        # Global features from attended regions
        global_feat = self.global_features(attended_features)
        
        # Flatten features
        scale3_feat = x3.view(x3.size(0), -1)
        global_feat = global_feat.view(global_feat.size(0), -1)
        
        # Feature fusion
        fused_features = torch.cat([scale3_feat, global_feat], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Regression predictions
        scores = self.regression_head(fused_features)
        scores = self.sigmoid(scores)
        
        return scores


class EfficientPalmCNNPredictor(nn.Module):
    """EfficientNet-based CNN for palm score regression"""
    
    def __init__(self, in_channels=1, backbone='efficientnet_b0'):
        super(EfficientPalmCNNPredictor, self).__init__()
        
        # Load pretrained EfficientNet
        if backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(backbone, 
                                            pretrained=True,
                                            in_chans=in_channels,
                                            num_classes=0)
            
            # Get feature dimension
            backbone_features = self.backbone(torch.randn(1, in_channels, 224, 224))
            feature_dim = backbone_features.shape[1]
        else:
            # Fallback to ResNet
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                          stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 512
        
        # Shared features
        self.shared_features = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Flatten features
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Process through shared layers
        shared_feat = self.shared_features(features)
        
        # Regression predictions
        scores = self.regression_head(shared_feat)
        scores = self.sigmoid(scores)
        
        return scores


class RegionAwarePalmCNNPredictor(nn.Module):
    """CNN with region-aware feature extraction for palm score regression"""
    
    def __init__(self, in_channels=1):
        super(RegionAwarePalmCNNPredictor, self).__init__()
        
        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Region-specific feature extractors
        self.region_extractors = nn.ModuleList([
            self._build_region_extractor(128, 64) for _ in range(4)
        ])
        
        # Attention modules
        self.attention_modules = nn.ModuleList([
            self._build_attention(128) for _ in range(4)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def _build_region_extractor(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _build_attention(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract shared features
        shared_feat = self.shared_backbone(x)
        
        region_features = []
        for i in range(4):
            # Apply attention
            attn = self.attention_modules[i](shared_feat)
            attended_features = shared_feat * attn
            
            # Extract region-specific features
            region_feat = self.region_extractors[i](attended_features)
            region_feat = region_feat.view(region_feat.size(0), -1)
            region_features.append(region_feat)
        
        # Concatenate all region features
        combined = torch.cat(region_features, dim=1)
        
        # Feature fusion
        fused = self.fusion(combined)
        
        # Regression predictions
        scores = self.regression_head(fused)
        scores = self.sigmoid(scores)
        
        return scores


def load_labels_from_json(json_path):
    """Load labels from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels_dict = {}
    for item in data:
        img_name = os.path.basename(item['image'])
        scores = json.loads(item['scores'])  # Parse the string JSON
        labels_dict[img_name] = scores

    return labels_dict


class PalmScorePipelineCNN:
    def __init__(self, model_type='multiscale', model_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model based on type
        if model_type == 'multiscale':
            self.model = MultiScaleCNNPredictor(in_channels=1).to(self.device)
        elif model_type == 'efficient':
            self.model = EfficientPalmCNNPredictor(in_channels=1).to(self.device)
        elif model_type == 'region':
            self.model = RegionAwarePalmCNNPredictor(in_channels=1).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.processor = PalmLineProcessor()
        
        # Image transformations for grayscale
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            print("Model loaded successfully!")
    
    def train(self, image_dir, labels_json_path, epochs=50, batch_size=8, lr=1e-4):
        """Train the CNN model on palm images with real labels"""
        print("🚀 Starting training process...")
        
        try:
            # Load labels
            labels_dict = load_labels_from_json(labels_json_path)
            print(f"📊 Loaded labels for {len(labels_dict)} images")
            
            # Load image paths
            image_dir_path = Path(image_dir)
            image_paths = list(image_dir_path.glob('*.jpg')) + \
                         list(image_dir_path.glob('*.png')) + \
                         list(image_dir_path.glob('*.jpeg'))
            
            if not image_paths:
                raise ValueError(f"No images found in {image_dir}")
            
            print(f"📁 Found {len(image_paths)} images")
            
            # Filter images that have labels
            valid_image_paths = []
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                if img_name in labels_dict:
                    valid_image_paths.append(img_path)
            
            print(f"📈 Images with labels: {len(valid_image_paths)}")
            
            if len(valid_image_paths) == 0:
                raise ValueError("No images with corresponding labels found!")
            
            # Split data
            train_paths, val_paths = train_test_split(
                valid_image_paths, test_size=0.2, random_state=42, shuffle=True
            )
            print(f"📊 Train/Val split: {len(train_paths)}/{len(val_paths)}")
            
            # Create datasets
            train_dataset = PalmScoreDataset(
                train_paths, labels_dict, transform=self.transform, is_training=True
            )
            val_dataset = PalmScoreDataset(
                val_paths, labels_dict, transform=self.transform, is_training=True
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, num_workers=0
            )
            
            # Optimizer and loss
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=0.01
            )
            criterion = nn.MSELoss()  # Use MSE for regression
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            print("🎯 Starting training loop...")
            
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                batch_count = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    predictions = self.model(images)
                    
                    # Calculate loss
                    loss = criterion(predictions, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 5 == 0:
                        print(
                            f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                # Average training loss
                avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
                train_losses.append(avg_train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        predictions = self.model(images)
                        loss = criterion(predictions, labels)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                print(
                    f'✅ Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model('palm_cnn_score_best.pth')
                    print(
                        f"💾 New best model saved with val loss: {best_val_loss:.4f}")
                
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_model(f'palm_cnn_score_epoch_{epoch+1}.pth')
            
            # Save final model
            self.save_model('palm_cnn_score_final.pth')
            print("🏁 Training completed!")
            
            # Print training summary
            print(f"\n📊 Training Summary:")
            print(f"   Final Train Loss: {train_losses[-1]:.4f}")
            print(f"   Final Val Loss: {val_losses[-1]:.4f}")
            print(f"   Best Val Loss: {best_val_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_scores(self, image_path):
        """Predict scores for a single palm image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        original_image = Image.open(image_path).convert('RGB')
        processed_image = self.processor.preprocess_image(original_image)
        input_tensor = self.transform(
            processed_image).unsqueeze(0).to(self.device)
        
        # Model prediction
        self.model.eval()
        with torch.no_grad():
            scores = self.model(input_tensor)
        
        # Convert to dictionary
        score_dict = {
            'strength': float(scores[0][0].cpu().numpy()),
            'romantic': float(scores[0][1].cpu().numpy()),
            'luck': float(scores[0][2].cpu().numpy()),
            'potential': float(scores[0][3].cpu().numpy())
        }
        
        return {
            'image_path': image_path,
            'scores': score_dict,
            'processed_image': processed_image
        }
    
    def evaluate_model(self, image_dir, labels_json_path):
        """Evaluate model on test set"""
        # Load labels
        labels_dict = load_labels_from_json(labels_json_path)
        
        # Get all images
        image_dir_path = Path(image_dir)
        image_paths = list(image_dir_path.glob('*.jpg')) + \
                     list(image_dir_path.glob('*.png'))
        
        # Filter images with labels
        test_paths = []
        true_labels = []
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            if img_name in labels_dict:
                test_paths.append(img_path)
                scores = labels_dict[img_name]
                true_labels.append([
                    scores.get('strength', 0.5),
                    scores.get('romantic', 0.5),
                    scores.get('luck', 0.5),
                    scores.get('potential', 0.5)
                ])
        
        print(f"🧪 Evaluating on {len(test_paths)} images...")
        
        predictions = []
        mse_loss = nn.MSELoss()
        total_loss = 0
        
        for i, img_path in enumerate(test_paths):
            result = self.predict_scores(str(img_path))
            pred_scores = list(result['scores'].values())
            true_scores = true_labels[i]
            
            # Calculate loss
            pred_tensor = torch.tensor(pred_scores, dtype=torch.float32)
            true_tensor = torch.tensor(true_scores, dtype=torch.float32)
            loss = mse_loss(pred_tensor, true_tensor)
            total_loss += loss.item()
            
            predictions.append({
                'image': os.path.basename(img_path),
                'predicted': result['scores'],
                'true': {
                    'strength': true_scores[0],
                    'romantic': true_scores[1],
                    'luck': true_scores[2],
                    'potential': true_scores[3]
                },
                'mse': loss.item()
            })
        
        avg_loss = total_loss / len(test_paths) if test_paths else 0
        print(f"📊 Average MSE Loss: {avg_loss:.4f}")
        
        return predictions, avg_loss
    
    def print_prediction_results(self, predictions):
        """Print prediction results in a readable format"""
        print("\n" + "="*60)
        print("🔮 PALM SCORE PREDICTION RESULTS (CNN)")
        print("="*60)
        
        for i, pred in enumerate(predictions):
            print(f"\n📊 Sample {i+1}: {pred['image']}")
            print("-" * 40)
            
            for score_type in ['strength', 'romantic', 'luck', 'potential']:
                pred_val = pred['predicted'][score_type]
                true_val = pred['true'][score_type]
                diff = abs(pred_val - true_val)
                print(f"  {score_type}: {pred_val:.3f} (pred) vs {true_val:.3f} (true) | diff: {diff:.3f}")
            
            print(f"  MSE: {pred['mse']:.4f}")
    
    def save_model(self, path):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(
                path, map_location=self.device))
            print(f"✅ Model loaded from {path}")
        else:
            print(f"❌ Model file not found: {path}")


def main():
    """Main function to run CNN-based palm score prediction"""
    
    print("="*60)
    print("🤖 CNN-BASED PALM SCORE PREDICTION SYSTEM")
    print("="*60)
    
    # Choose model type
    model_types = ['multiscale', 'efficient', 'region']
    print("\nAvailable CNN architectures:")
    for i, mt in enumerate(model_types, 1):
        print(f"  {i}. {mt}")
    
    choice = input("\nSelect model architecture (1-3, default=1): ").strip()
    if choice == '2':
        model_type = 'efficient'
    elif choice == '3':
        model_type = 'region'
    else:
        model_type = 'multiscale'
    
    print(f"\nUsing {model_type} CNN architecture")
    
    # Initialize pipeline
    pipeline = PalmScorePipelineCNN(model_type=model_type)
    
    # Train on your palm images with real labels
    print("\n" + "="*60)
    print("🎓 TRAINING PHASE")
    print("="*60)
    
    pipeline.train(
        image_dir='./data/resized_dataset',
        labels_json_path='./data/labels.json',
        epochs=20,
        batch_size=8,
        lr=1e-4
    )
    
    # Evaluate the model
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    predictions, avg_loss = pipeline.evaluate_model(
        './data/resized_dataset',
        './data/labels.json'
    )
    
    # Print sample predictions
    pipeline.print_prediction_results(predictions[:3])
    
    # Test inference on sample images
    print("\n" + "="*60)
    print("🎯 INFERENCE TESTING")
    print("="*60)
    
    # Test on a few sample images
    image_dir = './data/resized_dataset'
    image_paths = list(Path(image_dir).glob('*.jpg')) + \
                 list(Path(image_dir).glob('*.png'))
    
    if image_paths:
        test_paths = image_paths[:3]
        for img_path in test_paths:
            try:
                result = pipeline.predict_scores(str(img_path))
                print(f"\n📁 Image: {os.path.basename(img_path)}")
                for score_type, score in result['scores'].items():
                    print(f"  {score_type}: {score:.3f}")
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()