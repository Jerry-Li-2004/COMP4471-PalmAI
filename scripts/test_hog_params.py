import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.pipeline import PreprocessingPipeline

def test_hog_parameters():
    image_path = 'data/MALE/IMG_0001.JPG'
    output_dir = 'output/hog_tuning'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Read and resize image (using the same logic as the main script)
    image = cv2.imread(image_path)
    max_dim = 512
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    # Save original resized for reference
    cv2.imwrite(os.path.join(output_dir, 'original_resized.jpg'), image)

    # Define parameter sets to test
    # We want to capture fine lines, so we should decrease pixels_per_cell
    params_to_test = [
        {'name': 'baseline', 'ppc': (8, 8), 'cpb': (2, 2), 'orient': 9},
        {'name': 'finer_detail', 'ppc': (4, 4), 'cpb': (2, 2), 'orient': 9},
        {'name': 'very_fine', 'ppc': (2, 2), 'cpb': (2, 2), 'orient': 9},
        {'name': 'more_orientations', 'ppc': (4, 4), 'cpb': (2, 2), 'orient': 12},
    ]

    for params in params_to_test:
        print(f"Testing {params['name']}...")
        
        pipeline = PreprocessingPipeline(
            background_method='grabcut',
            grayscale_method='weighted',
            hog_orientations=params['orient'],
            hog_pixels_per_cell=params['ppc'],
            hog_cells_per_block=params['cpb']
        )
        
        # Process
        results = pipeline.process(image, return_intermediates=True)
        intermediates = results.get('intermediates', {})
        hog_image = intermediates.get('hog_image')
        
        if hog_image is not None:
            # Normalize for visualization
            hog_vis = ((hog_image - hog_image.min()) / 
                       (hog_image.max() - hog_image.min()) * 255).astype(np.uint8)
            
            filename = f"hog_{params['name']}_ppc{params['ppc'][0]}_orient{params['orient']}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, hog_vis)
            print(f"Saved {save_path}")

if __name__ == "__main__":
    test_hog_parameters()
