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


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H, W)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class MultiHeadPalmTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.n_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(
                embed_dim * mlp_ratio),
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)

        # Multi-task heads
        self.heart_line_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(
                0.2), nn.Linear(256, 10)
        )
        self.head_line_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(
                0.2), nn.Linear(256, 10)
        )
        self.life_line_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(
                0.2), nn.Linear(256, 15)
        )
        self.fate_line_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(
            ), nn.Dropout(0.2), nn.Linear(256, 8)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.ln(x[:, 0])

        return {
            'heart_line': self.heart_line_head(x),
            'head_line': self.head_line_head(x),
            'life_line': self.life_line_head(x),
            'fate_line': self.fate_line_head(x)
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


class PalmAnalysisPipeline:
    def __init__(self, model_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = MultiHeadPalmTransformer(in_channels=1).to(self.device)
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

    def train(self, image_dir, epochs=50, batch_size=8):
        """Train the model on palm images with better error handling"""
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

            # Test one sample
            test_sample, test_labels = train_dataset[0]
            print(f"🔍 Sample shape: {test_sample.shape}")

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, num_workers=0)

            # Optimizer and loss
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()

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

                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                val_losses.append(avg_val_loss)

                print(
                    f'✅ Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model('palm_transformer_best.pth')
                    print(
                        f"💾 New best model saved with val loss: {best_val_loss:.4f}")

                # Save model every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_model(f'palm_transformer_epoch_{epoch+1}.pth')

            # Save final model after training completes
            self.save_model('palm_transformer_final.pth')
            print(
                "🏁 Training completed! Final model saved as 'palm_transformer_final.pth'")

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
        print("🔮 PALM READING ANALYSIS RESULTS")
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


class PalmScorePredictor(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.n_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)

        # Score prediction heads (4 continuous scores)
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            # 4 output scores: strength, romantic, luck, potential
            nn.Linear(128, 4)
        )

        # Sigmoid activation for scores between 0-1
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.ln(x[:, 0])  # Use [CLS] token

        # Get scores
        scores = self.regression_head(x)
        scores = self.sigmoid(scores)  # Output between 0-1

        return scores


class PalmScoreDataset(Dataset):
    def __init__(self, image_paths, labels_dict=None, transform=None, is_training=True):
        self.image_paths = image_paths
        self.transform = transform
        self.is_training = is_training
        self.processor = PalmLineProcessor()

        # Load labels from JSON file
        self.labels = {}
        if labels_dict is not None:
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                if img_name in labels_dict:
                    scores = labels_dict[img_name]
                    # Convert to tensor
                    self.labels[img_name] = torch.tensor([
                        scores.get('strength', 0.5),
                        scores.get('romantic', 0.5),
                        scores.get('luck', 0.5),
                        scores.get('potential', 0.5)
                    ], dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')

            # Preprocess
            processed_image = self.processor.preprocess_image(image)

            if self.transform:
                processed_image = self.transform(processed_image)

            if self.is_training and self.labels:
                img_name = os.path.basename(image_path)
                if img_name in self.labels:
                    label = self.labels[img_name]
                else:
                    # Use default scores if not found
                    label = torch.tensor(
                        [0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
                return processed_image, label

            return processed_image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            dummy_image = torch.zeros(
                1, 224, 224) if self.transform else Image.new('L', (224, 224))
            if self.is_training:
                dummy_label = torch.tensor(
                    [0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
                return dummy_image, dummy_label
            return dummy_image


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


class PalmScorePipeline:
    def __init__(self, model_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = PalmScorePredictor(in_channels=1).to(self.device)
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
        """Train the model on palm images with real labels"""
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
                    self.save_model('palm_score_best.pth')
                    print(
                        f"💾 New best model saved with val loss: {best_val_loss:.4f}")

                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.save_model(f'palm_score_epoch_{epoch+1}.pth')

            # Save final model
            self.save_model('palm_score_final.pth')
            print("🏁 Training completed!")

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

    def save_model(self, path):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            self.model.load_state_dict(
                torch.load(path, map_location=self.device))
            print(f"✅ Model loaded from {path}")
        else:
            print(f"❌ Model file not found: {path}")


def main():
    # Initialize pipeline
    pipeline = PalmScorePipeline()

    # Train on your palm images with actual labels
    pipeline.train(
        image_dir='./data/resized_dataset',
        labels_json_path='./data/labels.json',  # Your JSON file
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
    print(f"\n📈 Sample Predictions (first 3):")
    print("-" * 60)
    for i, pred in enumerate(predictions[:3]):
        print(f"\nImage: {pred['image']}")
        print(f"MSE Loss: {pred['mse']:.4f}")
        print("Predicted vs True:")
        for score_type in ['strength', 'romantic', 'luck', 'potential']:
            pred_val = pred['predicted'][score_type]
            true_val = pred['true'][score_type]
            diff = abs(pred_val - true_val)
            print(
                f"  {score_type}: {pred_val:.3f} (pred) vs {true_val:.3f} (true) | diff: {diff:.3f}")


if __name__ == "__main__":
    main()
