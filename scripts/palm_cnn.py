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


class PalmLineDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.processor = PalmLineProcessor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess to enhance palm lines (convert to grayscale)
            processed_image = self.processor.preprocess_image(image)
            
            if self.transform:
                processed_image = self.transform(processed_image)
            
            if self.is_training and self.labels is not None:
                # Get labels for this specific sample
                sample_labels = {}
                for label_type, label_tensor in self.labels.items():
                    sample_labels[label_type] = label_tensor[idx]
                return processed_image, sample_labels
            
            return processed_image
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if there's an error
            dummy_image = torch.zeros(
                1, 224, 224) if self.transform else Image.new('L', (224, 224))
            if self.is_training and self.labels is not None:
                dummy_labels = {label_type: torch.tensor(
                    0) for label_type in self.labels.keys()}
                return dummy_image, dummy_labels
            return dummy_image


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for palm line feature extraction"""
    
    def __init__(self, in_channels=1):
        super(MultiScaleCNN, self).__init__()
        
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
        
        # Attention mechanism for important regions
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
        
        # Multi-task heads
        self.heart_line_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
        self.head_line_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
        self.life_line_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 15)
        )
        self.fate_line_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)
        )
        
        # Initialize weights
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
        
        # Multi-task predictions
        return {
            'heart_line': self.heart_line_head(fused_features),
            'head_line': self.head_line_head(fused_features),
            'life_line': self.life_line_head(fused_features),
            'fate_line': self.fate_line_head(fused_features)
        }


class EfficientPalmCNN(nn.Module):
    """EfficientNet-based CNN for palm line analysis"""
    
    def __init__(self, in_channels=1, backbone='efficientnet_b0'):
        super(EfficientPalmCNN, self).__init__()
        
        # Load pretrained EfficientNet
        if backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(backbone, 
                                            pretrained=True,
                                            in_chans=in_channels,
                                            num_classes=0)  # Remove classification head
            
            # Get feature dimension
            backbone_features = self.backbone(torch.randn(1, in_channels, 224, 224))
            feature_dim = backbone_features.shape[1]
        else:
            # Fallback to ResNet
            self.backbone = models.resnet18(pretrained=True)
            # Modify first conv layer for single channel input
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                          stride=2, padding=3, bias=False)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 512
        
        # Multi-task heads with shared backbone features
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
        
        # Task-specific heads
        self.heart_line_head = nn.Linear(256, 10)
        self.head_line_head = nn.Linear(256, 10)
        self.life_line_head = nn.Linear(256, 15)
        self.fate_line_head = nn.Linear(256, 8)
        
        # Line attention module
        self.line_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Flatten features
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Process through shared layers
        shared_feat = self.shared_features(features)
        
        # Multi-task predictions
        return {
            'heart_line': self.heart_line_head(shared_feat),
            'head_line': self.head_line_head(shared_feat),
            'life_line': self.life_line_head(shared_feat),
            'fate_line': self.fate_line_head(shared_feat)
        }


class RegionAwarePalmCNN(nn.Module):
    """CNN with region-aware feature extraction for palm lines"""
    
    def __init__(self, in_channels=1):
        super(RegionAwarePalmCNN, self).__init__()
        
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
        self.heart_line_extractor = self._build_region_extractor(128, 64)
        self.head_line_extractor = self._build_region_extractor(128, 64)
        self.life_line_extractor = self._build_region_extractor(128, 64)
        self.fate_line_extractor = self._build_region_extractor(128, 64)
        
        # Attention for each region
        self.heart_attention = self._build_attention(128)
        self.head_attention = self._build_attention(128)
        self.life_attention = self._build_attention(128)
        self.fate_attention = self._build_attention(128)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Task heads
        self.heart_line_head = nn.Linear(256, 10)
        self.head_line_head = nn.Linear(256, 10)
        self.life_line_head = nn.Linear(256, 15)
        self.fate_line_head = nn.Linear(256, 8)
    
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
        
        # Apply attention for each region
        heart_attn = self.heart_attention(shared_feat)
        head_attn = self.head_attention(shared_feat)
        life_attn = self.life_attention(shared_feat)
        fate_attn = self.fate_attention(shared_feat)
        
        # Extract region-specific features
        heart_feat = self.heart_line_extractor(shared_feat * heart_attn)
        head_feat = self.head_line_extractor(shared_feat * head_attn)
        life_feat = self.life_line_extractor(shared_feat * life_attn)
        fate_feat = self.fate_line_extractor(shared_feat * fate_attn)
        
        # Flatten and concatenate
        heart_feat = heart_feat.view(heart_feat.size(0), -1)
        head_feat = head_feat.view(head_feat.size(0), -1)
        life_feat = life_feat.view(life_feat.size(0), -1)
        fate_feat = fate_feat.view(fate_feat.size(0), -1)
        
        combined = torch.cat([heart_feat, head_feat, life_feat, fate_feat], dim=1)
        
        # Feature fusion
        fused = self.fusion(combined)
        
        # Multi-task predictions
        return {
            'heart_line': self.heart_line_head(fused),
            'head_line': self.head_line_head(fused),
            'life_line': self.life_line_head(fused),
            'fate_line': self.fate_line_head(fused)
        }


class PalmLineInterpreter:
    """Interprets model predictions based on palmistry rules"""
    
    def __init__(self):
        self.heart_line_interpretations = [
            "Content with love life",
            "Selfish when it comes to love",
            "Caring and understanding",
            "Less interest in romance",
            "Heart is broken easily",
            "Freely expresses emotions and feelings",
            "Good handle on emotions",
            "Many relationships, absence of serious relationships",
            "Sad or depressed",
            "Emotional trauma"
        ]
        
        self.head_line_interpretations = [
            "Prefers physical achievements over mental ones",
            "Creativity",
            "Inclination towards literature and fantasy",
            "Aptitude for math, business, and logic",
            "Adventure, enthusiasm for life",
            "Short attention span",
            "Thinking is clear and focused",
            "Thinks realistically",
            "Inconsistencies in thought or varying interests",
            "Momentous decisions"
        ]
        
        self.life_line_interpretations = [
            "Often tired",
            "Good physical and mental health",
            "Positive attitude towards life",
            "Pessimist",
            "Plenty of energy",
            "Enthusiastic and courageous",
            "Vitality",
            "Manipulated by others",
            "Strength and enthusiasm",
            "Cautious when it comes to relationships",
            "Academic achievement",
            "Success in business",
            "Sign of wealth",
            "Strong attachment with family",
            "Extra vitality"
        ]
        
        self.fate_line_interpretations = [
            "Strongly controlled by fate",
            "Successful life ahead",
            "Prone to many changes in life from external forces",
            "Great amount of wealth ahead",
            "Self-made individual; develops aspirations early on",
            "Interests must be surrendered to those of others",
            "Support offered by family and friends",
            "Comfortable but uneventful life ahead"
        ]
    
    def interpret_predictions(self, predictions, threshold=0.1):
        """Convert model predictions to human-readable interpretations"""
        interpretations = {}
        
        with torch.no_grad():
            # Heart line interpretation
            heart_probs = torch.softmax(predictions['heart_line'], dim=1)
            top_heart = torch.topk(heart_probs, 3, dim=1)
            interpretations['heart_line'] = [
                (self.heart_line_interpretations[i], f"{p:.2%}")
                for i, p in zip(top_heart.indices[0].cpu().numpy(),
                              top_heart.values[0].cpu().numpy())
                if p >= threshold
            ]
            
            # Head line interpretation
            head_probs = torch.softmax(predictions['head_line'], dim=1)
            top_head = torch.topk(head_probs, 3, dim=1)
            interpretations['head_line'] = [
                (self.head_line_interpretations[i], f"{p:.2%}")
                for i, p in zip(top_head.indices[0].cpu().numpy(),
                              top_head.values[0].cpu().numpy())
                if p >= threshold
            ]
            
            # Life line interpretation
            life_probs = torch.softmax(predictions['life_line'], dim=1)
            top_life = torch.topk(life_probs, 3, dim=1)
            interpretations['life_line'] = [
                (self.life_line_interpretations[i], f"{p:.2%}")
                for i, p in zip(top_life.indices[0].cpu().numpy(),
                              top_life.values[0].cpu().numpy())
                if p >= threshold
            ]
            
            # Fate line interpretation
            fate_probs = torch.softmax(predictions['fate_line'], dim=1)
            top_fate = torch.topk(fate_probs, 3, dim=1)
            interpretations['fate_line'] = [
                (self.fate_line_interpretations[i], f"{p:.2%}")
                for i, p in zip(top_fate.indices[0].cpu().numpy(),
                              top_fate.values[0].cpu().numpy())
                if p >= threshold
            ]
        
        return interpretations


class PalmAnalysisPipelineCNN:
    def __init__(self, model_type='multiscale', model_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model based on type
        if model_type == 'multiscale':
            self.model = MultiScaleCNN(in_channels=1).to(self.device)
        elif model_type == 'efficient':
            self.model = EfficientPalmCNN(in_channels=1).to(self.device)
        elif model_type == 'region':
            self.model = RegionAwarePalmCNN(in_channels=1).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.processor = PalmLineProcessor()
        self.interpreter = PalmLineInterpreter()
        
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
    
    def create_dummy_labels(self, num_samples):
        """Create proper dummy labels"""
        return {
            'heart_line': torch.randint(0, 10, (num_samples,)),
            'head_line': torch.randint(0, 10, (num_samples,)),
            'life_line': torch.randint(0, 15, (num_samples,)),
            'fate_line': torch.randint(0, 8, (num_samples,))
        }
    
    def train(self, image_dir, epochs=50, batch_size=8, lr=1e-4):
        """Train the CNN model on palm images"""
        print("🚀 Starting training process...")
        
        try:
            # Load and prepare data
            image_dir_path = Path(image_dir)
            if not image_dir_path.exists():
                raise ValueError(f"Image directory not found: {image_dir}")
            
            image_paths = list(image_dir_path.glob('*.jpg')) + \
                list(image_dir_path.glob('*.png'))
            
            if not image_paths:
                # Try with different extensions
                image_paths = list(image_dir_path.glob(
                    '*.jpeg')) + list(image_dir_path.glob('*.JPG'))
            
            if not image_paths:
                raise ValueError(
                    f"No images found in {image_dir}. Supported formats: jpg, png, jpeg, JPG")
            
            print(f"📁 Found {len(image_paths)} images")
            print(
                f"📸 Sample images: {[path.name for path in image_paths[:3]]}")
            
            # Split data
            train_paths, val_paths = train_test_split(
                image_paths, test_size=0.2, random_state=42)
            print(f"📊 Train/Val split: {len(train_paths)}/{len(val_paths)}")
            
            # Create dummy labels
            train_labels = self.create_dummy_labels(len(train_paths))
            val_labels = self.create_dummy_labels(len(val_paths))
            
            # Create datasets
            train_dataset = PalmLineDataset(
                train_paths, train_labels, transform=self.transform)
            val_dataset = PalmLineDataset(
                val_paths, val_labels, transform=self.transform)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, num_workers=0)
            
            # Optimizer and loss
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2)
            
            print("🎯 Starting training loop...")
            
            # Training variables
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            # Training loop
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                batch_count = 0
                
                for batch_idx, batch_data in enumerate(train_loader):
                    # Unpack the data correctly
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, labels = batch_data
                    else:
                        print(
                            f"❌ Unexpected batch data format: {type(batch_data)}")
                        continue
                    
                    # Move to device
                    images = images.to(self.device)
                    labels = {k: v.to(self.device) for k, v in labels.items()}
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate multi-task loss
                    loss = (criterion(outputs['heart_line'], labels['heart_line']) +
                          criterion(outputs['head_line'], labels['head_line']) +
                          criterion(outputs['life_line'], labels['life_line']) +
                          criterion(outputs['fate_line'], labels['fate_line'])) / 4
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 5 == 0:  # Print more frequently
                        print(
                            f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                if batch_count > 0:
                    avg_train_loss = total_loss / batch_count
                    train_losses.append(avg_train_loss)
                else:
                    avg_train_loss = 0
                    train_losses.append(0)
                    print("⚠️  No batches processed this epoch")
                
                # Validation
                self.model.eval()
                val_loss = 0
                val_batches = 0
                val_correct = {k: 0 for k in ['heart', 'head', 'life', 'fate']}
                val_total = 0
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            images, labels = batch_data
                            images = images.to(self.device)
                            labels = {k: v.to(self.device)
                                    for k, v in labels.items()}
                            
                            outputs = self.model(images)
                            loss = (criterion(outputs['heart_line'], labels['heart_line']) +
                                  criterion(outputs['head_line'], labels['head_line']) +
                                  criterion(outputs['life_line'], labels['life_line']) +
                                  criterion(outputs['fate_line'], labels['fate_line'])) / 4
                            val_loss += loss.item()
                            val_batches += 1
                            
                            # Calculate accuracy for each task
                            for task, output in outputs.items():
                                preds = output.argmax(dim=1)
                                correct = (preds == labels[task]).sum().item()
                                val_correct[task.split('_')[0]] += correct
                                val_total += labels[task].size(0)
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                val_losses.append(avg_val_loss)
                
                # Calculate accuracy
                accuracies = {k: (v / (val_total / 4) * 100) if val_total > 0 else 0 
                            for k, v in val_correct.items()}
                
                print(f'✅ Epoch {epoch+1}/{epochs}, '
                    f'Train Loss: {avg_train_loss:.4f}, '
                    f'Val Loss: {avg_val_loss:.4f}')
                print(f'   Accuracies: Heart: {accuracies["heart"]:.1f}%, '
                    f'Head: {accuracies["head"]:.1f}%, '
                    f'Life: {accuracies["life"]:.1f}%, '
                    f'Fate: {accuracies["fate"]:.1f}%')
                
                # Update learning rate
                scheduler.step()
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model('palm_cnn_best.pth')
                    print(
                        f"💾 New best model saved with val loss: {best_val_loss:.4f}")
                
                # Save model every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_model(f'palm_cnn_epoch_{epoch+1}.pth')
            
            # Save final model
            self.save_model('palm_cnn_final.pth')
            print(
                "🏁 Training completed! Final model saved as 'palm_cnn_final.pth'")
            
            # Print training summary
            print(f"\n📊 Training Summary:")
            print(f"   Final Train Loss: {train_losses[-1]:.4f}")
            print(f"   Final Val Loss: {val_losses[-1]:.4f}")
            print(f"   Best Val Loss: {best_val_loss:.4f}")
        
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_palm(self, image_path):
        """Analyze a single palm image with detailed interpretations"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"🔍 Analyzing: {Path(image_path).name}")
        
        # Preprocess image
        original_image = Image.open(image_path).convert('RGB')
        processed_image = self.processor.preprocess_image(original_image)
        input_tensor = self.transform(
            processed_image).unsqueeze(0).to(self.device)
        
        # Model prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Get interpretations
        interpretations = self.interpreter.interpret_predictions(predictions)
        
        return {
            'image_path': image_path,
            'predictions': predictions,
            'interpretations': interpretations,
            'processed_image': processed_image
        }
    
    def print_analysis_results(self, analysis_result):
        """Print analysis results in a readable format"""
        result = analysis_result
        interpretations = result['interpretations']
        
        print("\n" + "="*60)
        print("🔮 PALM READING ANALYSIS RESULTS (CNN)")
        print("="*60)
        print(f"📁 Image: {Path(result['image_path']).name}")
        print("="*60)
        
        for line_type, line_interpretations in interpretations.items():
            if line_interpretations:  # Only print if there are interpretations
                line_name = line_type.replace('_', ' ').title()
                print(f"\n📖 {line_name}:")
                print("-" * 40)
                
                for i, (interpretation, confidence) in enumerate(line_interpretations):
                    rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"{rank_symbol} {interpretation}")
                    print(f"   Confidence: {confidence}")
                print()
    
    def test_inference(self, image_dir, num_samples=3):
        """Test inference on sample images from the training set"""
        print("\n🧪 TESTING INFERENCE ON TRAINING SAMPLES")
        print("="*60)
        
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
            list(Path(image_dir).glob('*.png'))
        
        if not image_paths:
            print("❌ No images found for testing")
            return
        
        # Select random samples for testing
        test_paths = image_paths[:min(num_samples, len(image_paths))]
        
        for i, image_path in enumerate(test_paths):
            print(f"\n📊 Sample {i+1}/{len(test_paths)}: {image_path.name}")
            print("-" * 40)
            
            try:
                result = self.analyze_palm(str(image_path))
                self.print_analysis_results(result)
            except Exception as e:
                print(f"❌ Error analyzing {image_path.name}: {e}")
    
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
    """Main function to run CNN-based palm analysis"""
    
    print("="*60)
    print("🤖 CNN-BASED PALM LINE ANALYSIS SYSTEM")
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
    pipeline = PalmAnalysisPipelineCNN(model_type=model_type)
    
    # Train on your palm images
    print("\n" + "="*60)
    print("🎓 TRAINING PHASE")
    print("="*60)
    
    pipeline.train(
        # image_dir='/Users/jerrylhm/Desktop/Fall 2025-26/COMP4471/COMP4471-PalmAI/output/resized_dataset',
        image_dir='/home/javan/Desktop/4471/project/COMP4471-PalmAI/data/resized_dataset',
        epochs=10,  # Can adjust based on needs
        batch_size=4,
        lr=1e-4
    )
    
    # Test inference on sample images
    print("\n" + "="*60)
    print("🎯 INFERENCE TESTING")
    print("="*60)
    
    pipeline.test_inference(
        # '/Users/jerrylhm/Desktop/Fall 2025-26/COMP4471/COMP4471-PalmAI/output/resized_dataset',
        '/home/javan/Desktop/4471/project/COMP4471-PalmAI/data/resized_dataset',
        num_samples=3
    )
    
    # Demonstrate loading the saved model and running inference
    print("\n" + "="*60)
    print("🔄 TESTING MODEL LOADING AND INFERENCE")
    print("="*60)
    
    # Create a new pipeline instance and load the trained model
    loaded_pipeline = PalmAnalysisPipelineCNN(model_type=model_type)
    loaded_pipeline.load_model('palm_cnn_final.pth')
    
    # Test inference with the loaded model
    loaded_pipeline.test_inference(
        # '/Users/jerrylhm/Desktop/Fall 2025-26/COMP4471/COMP4471-PalmAI/output/resized_dataset',
        '/home/javan/Desktop/4471/project/COMP4471-PalmAI/data/resized_dataset',
        num_samples=2
    )
    
    # Compare different model architectures
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    print("CNN architectures provided:")
    print("1. MultiScaleCNN: Multi-scale feature extraction with attention")
    print("2. EfficientPalmCNN: EfficientNet-based with pretrained weights")
    print("3. RegionAwarePalmCNN: Region-specific feature extraction")
    print("\n✅ Training complete! Models saved as:")
    print("   - palm_cnn_best.pth (best validation loss)")
    print("   - palm_cnn_final.pth (final trained model)")


if __name__ == "__main__":
    main()