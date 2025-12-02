import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.pipeline import PreprocessingPipeline

def process_dataset(data_dir, output_dir):
    """
    Process all images in the dataset (MALE and FEMALE folders).
    
    Args:
        data_dir: Path to the data directory containing MALE and FEMALE folders.
        output_dir: Path to save processed results.
    """
    
    # Initialize Pipeline
    # Updated HOG parameters for better palm line visibility (smaller cells)
    pipeline = PreprocessingPipeline(
        background_method='grabcut',
        grayscale_method='weighted',
        hog_orientations=9,
        hog_pixels_per_cell=(4, 4),
        hog_cells_per_block=(2, 2)
    )
    
    categories = ['MALE', 'FEMALE']
    
    # Limit for testing (Set to None for full dataset)
    LIMIT_PER_CATEGORY = None
    
    for category in categories:
        input_path = os.path.join(data_dir, category)
        output_path = os.path.join(output_dir, category)
        
        if not os.path.exists(input_path):
            print(f"Warning: Directory {input_path} does not exist. Skipping.")
            continue
            
        os.makedirs(output_path, exist_ok=True)
        
        # Get list of images
        images = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        # Sort to ensure consistent order
        images.sort()
        
        if LIMIT_PER_CATEGORY:
            images = images[:LIMIT_PER_CATEGORY]
            print(f"Processing first {LIMIT_PER_CATEGORY} images in {category} for testing...")
        else:
            print(f"Processing {len(images)} images in {category}...")
        
        for image_name in tqdm(images, desc=f"Processing {category}"):
            image_file = os.path.join(input_path, image_name)
            
            try:
                # Read image
                image = cv2.imread(image_file)
                if image is None:
                    print(f"Error reading {image_file}. Skipping.")
                    continue
                
                # Resize image for faster processing
                # GrabCut is very slow on high-res images. Resizing to a max dimension of 512 or 800 is recommended.
                max_dim = 512
                h, w = image.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                
                # Process image
                results = pipeline.process(image, return_intermediates=True)
                
                # Extract results
                intermediates = results.get('intermediates', {})
                foreground = intermediates.get('foreground')
                hog_image = intermediates.get('hog_image')
                enhanced_lines = intermediates.get('enhanced_lines')
                
                # Normalize HOG image for visualization/saving
                if hog_image is not None:
                    hog_vis = ((hog_image - hog_image.min()) / 
                               (hog_image.max() - hog_image.min()) * 255).astype(np.uint8)
                else:
                    hog_vis = np.zeros_like(results['final_image'])

                # Save results
                base_name = os.path.splitext(image_name)[0]
                
                # Save Background Extracted
                bg_extracted_path = os.path.join(output_path, f"{base_name}_bg_extracted.jpg")
                cv2.imwrite(bg_extracted_path, foreground)
                
                # Save Enhanced Lines (This is the key for training data)
                if enhanced_lines is not None:
                    lines_path = os.path.join(output_path, f"{base_name}_lines_enhanced.jpg")
                    cv2.imwrite(lines_path, enhanced_lines)
                
                # Save HOG Visualization
                hog_path = os.path.join(output_path, f"{base_name}_hog.jpg")
                cv2.imwrite(hog_path, hog_vis)
                
                # Optionally save the HOG features as .npy if needed for ML, 
                # but for now images are good for verification.
                # np.save(os.path.join(output_path, f"{base_name}_hog_features.npy"), results['hog_features'])
                
            except Exception as e:
                print(f"Failed to process {image_file}: {str(e)}")

if __name__ == "__main__":
    # Define paths
    # Assuming script is run from project root or scripts folder
    # We use absolute paths based on the file location to be safe
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_directory = os.path.join(project_root, 'data')
    output_directory = os.path.join(project_root, 'output', 'preprocessed_dataset')
    
    print(f"Data Directory: {data_directory}")
    print(f"Output Directory: {output_directory}")
    
    process_dataset(data_directory, output_directory)
    print("Dataset preprocessing complete!")
