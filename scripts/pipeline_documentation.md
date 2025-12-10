# Palm Image Analysis Pipeline Documentation

## 1. Executive Summary
The palm image analysis pipeline has been refactored to use **Transfer Learning** with a **ResNet18** backbone. This replaces the previous custom Vision Transformer implementation to improve training stability and convergence on the dataset (~800 images).

## 2. Model Architecture
### Backbone: ResNet18
- **Source**: `torchvision.models.resnet18`
- **Weights**: Pretrained on `ImageNet1K_V1`.
- **Freezing Strategy**:
  - **Layers 1-3**: Frozen (weights are static).
  - **Layer 4**: Unfrozen (trainable) to allow the model to adapt high-level features specific to palm lines.
  - **FC Layer**: Replaced with a custom regression head.

### Regression Head
The original 1000-class classification head is replaced with:
1.  **Dropout (0.5)**: Regularization.
2.  **Linear (512 -> 128)**: Dimensionality reduction.
3.  **ReLU**: Activation.
4.  **Dropout (0.5)**: Further regularization.
5.  **Linear (128 -> 4)**: Output layer for 4 scores (Strength, Romantic, Luck, Potential).
6.  **Sigmoid**: Activates outputs to the `[0, 1]` range.

## 3. Data Preprocessing
Standard ImageNet preprocessing is applied to ensure compatibility with the pretrained backbone.

### Training Transforms
- **Resize**: To 256x256.
- **RandomCrop**: To 224x224 (Data Augmentation).
- **RandomHorizontalFlip**: Data Augmentation.
- **RandomRotation**: +/- 15 degrees.
- **ColorJitter**: Brightness/Contrast variation (0.2).
- **Normalize**: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

### Validation Transforms
- **Resize**: To 224x224 directly.
- **Normalize**: Same as training.

## 4. Training Configuration
- **Loss Function**: `MSELoss` (Mean Squared Error) for regression.
- **Optimizer**: `Adam`
  - **Learning Rate**: `5e-5` (Low learning rate for fine-tuning).
  - **Weight Decay**: `1e-4` (L2 Regularization).
- **Scheduler**: `ReduceLROnPlateau` (Reduces LR by factor of 0.5 if validation loss stagnates for 3 epochs).
- **Hardware**: Automatically detects CUDA (GPU) or CPU (mps support can be added if needed).
- **SSL**: Auto-handling of macOS SSL certificate verification.

## 5. Rationale
1.  **Switch to ResNet**: Transformers (ViT) require large amounts of data to converge. For <1000 images, CNNs (like ResNet) are significantly more robust.
2.  **Freezing Early Layers**: Low-level features (edges, textures) are universal. Freezing them prevents overfitting and speeds up training.
3.  **Unfreezing Layer 4**: Palm lines are high-level textures. Unfreezing the final convolutional block allows the model to learn these specific patterns.
4.  **Dropout**: Heavy dropout (0.5) is used to prevent the model from memorizing the small training set.
