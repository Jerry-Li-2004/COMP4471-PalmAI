"""
Demonstration script for preprocessing pipeline.
Shows all preprocessing steps on sample images from the 11k Hands dataset.
"""
import cv2
import os
import sys
import numpy as np

# Add preprocessing directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../preprocessing'))

from pipeline import PreprocessingPipeline

def demo_preprocessing():
    """Demonstrate preprocessing pipeline on sample images."""
    
    print("=" * 80)
    print("Palm Image Preprocessing Pipeline Demo")
    print("=" * 80)
    
    # Setup paths
    data_dir = os.path.join(os.path.dirname(__file__), '../data/11k_hands/Hands/Hands')
    output_dir = os.path.join(os.path.dirname(__file__), '../output/preprocessing_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please ensure the 11k Hands dataset is downloaded.")
        return
    
    # Get sample images
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')][:5]
    
    if not image_files:
        print(f"\n❌ No images found in: {data_dir}")
        return
    
    print(f"\n✓ Found {len(image_files)} sample images")
    print(f"✓ Output directory: {output_dir}\n")
    
    # Initialize pipeline
    print("Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        background_method='grabcut',
        grayscale_method='weighted',
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        hog_cells_per_block=(2, 2)
    )
    print("✓ Pipeline initialized\n")
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_file}")
        
        # Load image
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  ❌ Failed to load image")
            continue
        
        print(f"  Image shape: {image.shape}")
        
        # Process through pipeline
        try:
            results = pipeline.process(image, return_intermediates=True)
            
            print(f"  ✓ Background extraction complete")
            print(f"  ✓ Grayscale conversion complete")
            print(f"  ✓ HOG feature extraction complete")
            print(f"  HOG feature vector shape: {results['hog_features'].shape}")
            
            # Create visualization
            vis_path = os.path.join(output_dir, f"preprocessing_{img_file}")
            vis = pipeline.visualize_pipeline(image, save_path=vis_path)
            
            print(f"  ✓ Visualization saved: {vis_path}\n")
            
        except Exception as e:
            print(f"  ❌ Error processing image: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print("Demo complete!")
    print(f"Check {output_dir} for visualization results.")
    print("=" * 80)

if __name__ == "__main__":
    demo_preprocessing()
