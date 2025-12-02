import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.line_identification import PalmLineIdentifier

def main():
    # Configuration
    image_paths = [
        'data/11k_hands/palmar_with_lines/Hand_0000654.jpg'
    ]
    output_base_dir = 'output/processed_palm_batch'
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Initialize Pipeline
    pipeline = PreprocessingPipeline(
        background_method='grabcut',
        grayscale_method='weighted',
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        hog_cells_per_block=(2, 2)
    )
    
    # Initialize Line Identifier
    line_identifier = PalmLineIdentifier()
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            continue

        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        image_name = os.path.basename(image_path).split('.')[0]
        output_dir = os.path.join(output_base_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(image_path)
        
        # Run Preprocessing Pipeline (BG -> Gray -> HOG)
        results = pipeline.process(image, return_intermediates=True)
        
        # Extract results
        gray_image = results['final_image']
        hog_features = results['hog_features']
        
        intermediates = results.get('intermediates', {})
        hog_image = intermediates.get('hog_image')
        mask = intermediates.get('background_mask') # GrabCut mask
        
        # Run Line Identification
        # Use the mask from background extraction for better ROI
        line_masks = line_identifier.extract_lines(gray_image, mask=mask, side='auto')
        
        # Visualize Lines
        lines_vis = line_identifier.visualize_lines(gray_image, line_masks)
        
        # Save Results
        print(f"Saving results to {output_dir}...")
        
        # 1. Original
        cv2.imwrite(os.path.join(output_dir, '01_original.jpg'), image)
        
        # 2. Background Extracted (Mask)
        if mask is not None:
            cv2.imwrite(os.path.join(output_dir, '02_mask.jpg'), mask)
            # Apply mask to original
            masked_img = cv2.bitwise_and(image, image, mask=mask)
            cv2.imwrite(os.path.join(output_dir, '02_bg_removed.jpg'), masked_img)
        else:
            masked_img = image # Fallback
            
        # 3. Grayscale
        cv2.imwrite(os.path.join(output_dir, '03_grayscale.jpg'), gray_image)
        
        # 4. HOG Visualization
        if hog_image is not None:
            hog_vis_norm = (hog_image / hog_image.max() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, '04_hog.jpg'), hog_vis_norm)
            
        # 5. Palm Lines
        cv2.imwrite(os.path.join(output_dir, '05_palm_lines.jpg'), lines_vis)
        
        # Create a composite visualization
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("Background Removed")
        if mask is not None:
            plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        else:
            plt.text(0.5, 0.5, "No Mask", ha='center')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("Grayscale")
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("HOG Features")
        plt.imshow(hog_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("Identified Lines")
        plt.imshow(cv2.cvtColor(lines_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pipeline_summary.png'))
        plt.close()
        
    print("Batch processing complete!")

if __name__ == "__main__":
    main()
