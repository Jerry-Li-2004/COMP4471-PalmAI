import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
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

# ==========================================
# 1. Dataset Class
# ==========================================


class PalmDataset(Dataset):
    """Dataset for loading palm images and their labels"""

    def __init__(self, image_paths, labels_dict, transform=None):
        self.image_paths = image_paths
        self.labels_dict = labels_dict
        self.transform = transform
        self.image_names = [os.path.basename(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        # Get Label
        scores = self.labels_dict.get(img_name, None)

        if scores is None:
            label_tensor = torch.tensor(
                [0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        else:
            label_tensor = torch.tensor([
                float(scores.get('strength', 0.5)),
                float(scores.get('romantic', 0.5)),
                float(scores.get('luck', 0.5)),
                float(scores.get('potential', 0.5))
            ], dtype=torch.float32)

        # Apply Transforms
        if self.transform:
            image = self.transform(image)

        return image, label_tensor, str(img_path), img_name

# ==========================================
# 2. Label Loading Function
# ==========================================


def load_labels(json_path):
    """Load and parse label JSON file"""
    print(f"Loading labels from {json_path}...")

    if not os.path.exists(json_path):
        print(f"ERROR: Labels file not found: {json_path}")
        return {}

    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    clean_labels = {}

    if isinstance(raw_data, list):
        iterable = raw_data
    elif isinstance(raw_data, dict):
        if all(key in raw_data for key in ['image', 'scores']):
            iterable = [raw_data]
        else:
            iterable = raw_data.values()
    else:
        print("Unknown JSON format")
        return {}

    count = 0
    for item in iterable:
        img_ref = item.get('image', '')
        if not img_ref:
            continue

        img_name = os.path.basename(img_ref)
        scores_obj = item.get('scores', {})

        if isinstance(scores_obj, str):
            try:
                scores_obj = json.loads(scores_obj)
            except json.JSONDecodeError:
                continue

        required_keys = ['strength', 'romantic', 'luck', 'potential']
        if all(k in scores_obj for k in required_keys):
            clean_scores = {}
            for key in required_keys:
                try:
                    clean_scores[key] = float(scores_obj[key])
                except ValueError:
                    clean_scores[key] = 0.5

            clean_labels[img_name] = clean_scores
            count += 1

    print(f"Successfully loaded {count} labeled entries.")
    return clean_labels

# ==========================================
# 3. Model Architecture
# ==========================================


class PalmResNet(nn.Module):
    """ResNet18 based model for palm reading predictions"""

    def __init__(self, pretrained=True, freeze_backbone=False):
        super(PalmResNet, self).__init__()
        self.backbone = models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 4. Main Evaluation Function
# ==========================================


def evaluate_all_images(model_path, data_dir, label_file, save_dir):
    """
    Main function to evaluate model on all images
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if files exist
    if not os.path.exists(label_file):
        print(f"ERROR: Labels file not found: {label_file}")
        return None

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return None

    # Load labels
    print("\n" + "="*60)
    print("LOADING LABELS")
    print("="*60)
    labels_dict = load_labels(label_file)

    if not labels_dict:
        print("ERROR: No labels loaded.")
        return None

    # Find all images
    print("\n" + "="*60)
    print("FINDING IMAGES")
    print("="*60)
    all_image_paths = []
    supported_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for ext in supported_exts:
        all_image_paths.extend(list(Path(data_dir).glob(f"*{ext}")))
        all_image_paths.extend(list(Path(data_dir).glob(f"*{ext.upper()}")))

    print(f"Found {len(all_image_paths)} total images in {data_dir}")

    # Filter only those with labels
    valid_paths = [
        str(p) for p in all_image_paths if os.path.basename(p) in labels_dict]
    print(f"Found {len(valid_paths)} images with labels")

    if len(valid_paths) == 0:
        print("ERROR: No labeled images found!")
        return None

    # Create dataset
    print("\n" + "="*60)
    print("CREATING DATASET")
    print("="*60)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = PalmDataset(valid_paths, labels_dict, transform=transform)

    # Create dataloader
    batch_size = 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"Created dataset with {len(dataset)} images")

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    print(f"Loading model from {model_path}...")
    model = PalmResNet(pretrained=False, freeze_backbone=False)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    all_predictions = []
    all_ground_truth = []
    all_image_names = []

    with torch.no_grad():
        for images, labels, paths, img_names in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            all_predictions.append(outputs.cpu().numpy())
            all_ground_truth.append(labels.numpy())
            all_image_names.extend(list(img_names))

    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_ground_truth = np.vstack(all_ground_truth)

    print(f"\nEvaluation complete!")
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Ground truth shape: {all_ground_truth.shape}")

    # Calculate metrics
    print("\n" + "="*60)
    print("CALCULATING METRICS")
    print("="*60)

    attribute_names = ['strength', 'romantic', 'luck', 'potential']

    # Calculate per-attribute metrics
    per_attribute_metrics = {}
    for i, attr in enumerate(attribute_names):
        attr_pred = all_predictions[:, i]
        attr_true = all_ground_truth[:, i]

        per_attribute_metrics[attr] = {
            'mse': mean_squared_error(attr_true, attr_pred),
            'mae': mean_absolute_error(attr_true, attr_pred),
            'r2': r2_score(attr_true, attr_pred),
            'correlation': np.corrcoef(attr_true, attr_pred)[0, 1],
            'mean_true': np.mean(attr_true),
            'mean_pred': np.mean(attr_pred)
        }

    # Calculate overall metrics
    overall_mse = mean_squared_error(
        all_ground_truth.flatten(), all_predictions.flatten())
    overall_mae = mean_absolute_error(
        all_ground_truth.flatten(), all_predictions.flatten())
    overall_r2 = r2_score(all_ground_truth.flatten(),
                          all_predictions.flatten())

    # Calculate per-image MSE
    per_image_mse = []
    per_image_results = []
    for i in range(len(all_ground_truth)):
        mse = mean_squared_error(all_ground_truth[i], all_predictions[i])
        mae = mean_absolute_error(all_ground_truth[i], all_predictions[i])
        per_image_mse.append(mse)

        per_image_results.append({
            'image_name': all_image_names[i],
            'mse': mse,
            'mae': mae,
            'strength_true': all_ground_truth[i, 0],
            'romantic_true': all_ground_truth[i, 1],
            'luck_true': all_ground_truth[i, 2],
            'potential_true': all_ground_truth[i, 3],
            'strength_pred': all_predictions[i, 0],
            'romantic_pred': all_predictions[i, 1],
            'luck_pred': all_predictions[i, 2],
            'potential_pred': all_predictions[i, 3]
        })

    per_image_mse = np.array(per_image_mse)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)

    print(f"\nOverall Performance:")
    print(f"  MSE: {overall_mse:.6f}")
    print(f"  MAE: {overall_mae:.6f}")
    print(f"  R²:  {overall_r2:.6f}")

    print(f"\nPer-Attribute Performance:")
    for attr in attribute_names:
        metrics = per_attribute_metrics[attr]
        print(f"\n{attr.upper():12s}:")
        print(f"  MSE:         {metrics['mse']:.6f}")
        print(f"  MAE:         {metrics['mae']:.6f}")
        print(f"  R²:          {metrics['r2']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.6f}")

    print(f"\nPer-image MSE Statistics:")
    print(
        f"  Mean ± Std: {per_image_mse.mean():.6f} ± {per_image_mse.std():.6f}")
    print(f"  Min:        {per_image_mse.min():.6f}")
    print(f"  Max:        {per_image_mse.max():.6f}")
    print(f"  Median:     {np.median(per_image_mse):.6f}")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save detailed results to CSV
    detailed_results = []
    for i, img_name in enumerate(all_image_names):
        row = {'image_name': img_name, 'mse': per_image_mse[i]}

        for j, attr in enumerate(attribute_names):
            row[f'{attr}_true'] = all_ground_truth[i, j]
            row[f'{attr}_pred'] = all_predictions[i, j]
            row[f'{attr}_error'] = all_ground_truth[i, j] - \
                all_predictions[i, j]

        detailed_results.append(row)

    df_detailed = pd.DataFrame(detailed_results)
    csv_path = os.path.join(save_dir, 'detailed_predictions.csv')
    df_detailed.to_csv(csv_path, index=False)
    print(f"Detailed predictions saved to: {csv_path}")

    # Save summary
    summary_path = os.path.join(save_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total images evaluated: {len(all_image_names)}\n")
        f.write(f"Overall MSE: {overall_mse:.6f}\n")
        f.write(f"Overall MAE: {overall_mae:.6f}\n")
        f.write(f"Overall R²: {overall_r2:.6f}\n\n")

        f.write("-"*60 + "\n")
        f.write("PER-ATTRIBUTE METRICS:\n")
        f.write("-"*60 + "\n")

        for attr in attribute_names:
            metrics = per_attribute_metrics[attr]
            f.write(f"\n{attr.upper()}:\n")
            f.write(f"  MSE:         {metrics['mse']:.6f}\n")
            f.write(f"  MAE:         {metrics['mae']:.6f}\n")
            f.write(f"  R²:          {metrics['r2']:.6f}\n")
            f.write(f"  Correlation: {metrics['correlation']:.6f}\n")

    print(f"Summary saved to: {summary_path}")

    # Save worst predictions
    sorted_indices = np.argsort(per_image_mse)[::-1][:10]

    worst_path = os.path.join(save_dir, 'worst_predictions.txt')
    with open(worst_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TOP 10 WORST PREDICTIONS (Highest MSE)\n")
        f.write("="*60 + "\n\n")

        for rank, idx in enumerate(sorted_indices, 1):
            f.write(f"\n{rank}. {all_image_names[idx]}\n")
            f.write(f"   MSE: {per_image_mse[idx]:.6f}\n")

    print(f"Worst predictions saved to: {worst_path}")

    # Create simple visualization
    plt.figure(figsize=(10, 6))
    plt.hist(per_image_mse, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(per_image_mse.mean(), color='red', linestyle='--',
                label=f'Mean: {per_image_mse.mean():.6f}')
    plt.xlabel('MSE per Image')
    plt.ylabel('Frequency')
    plt.title('Distribution of MSE per Image')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mse_distribution.png'), dpi=150)
    plt.close()

    print(
        f"Visualization saved to: {os.path.join(save_dir, 'mse_distribution.png')}")

    print(f"\nAll results saved to: {save_dir}")

    return {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'per_attribute_metrics': per_attribute_metrics,
        'per_image_mse': per_image_mse
    }

# ==========================================
# 5. Main Execution
# ==========================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate palm reading model on all images')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels JSON file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    print("="*60)
    print("PALM READING MODEL EVALUATION")
    print("="*60)

    # Run evaluation
    results = evaluate_all_images(
        model_path=os.path.abspath(args.model),
        data_dir=os.path.abspath(args.data_dir),
        label_file=os.path.abspath(args.labels),
        save_dir=os.path.abspath(args.output_dir)
    )

    if results:
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"\nOverall MSE: {results['overall_mse']:.6f}")
