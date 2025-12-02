import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.line_identification import PalmLineIdentifier

def organize_and_process_lines(input_dir, output_dir):
    """
    Selects 'lines_enhanced' images, rotates them, identifies hand side,
    and saves them to a new folder structure.
    
    Args:
        input_dir: Directory containing the preprocessed dataset (MALE/FEMALE subfolders).
        output_dir: Directory to save the organized images.
    """
    
    # Initialize Line Identifier for hand side detection
    # Note: Hand side detection usually works best on the binary mask or original image.
    # If we only have the line image, it might be tricky. 
    # Let's assume we can find the corresponding mask or use the line structure.
    # Actually, the user request implies we should identify it now.
    # The PalmLineIdentifier.detect_hand_side uses a binary mask.
    # We can try to generate a mask from the line image (non-zero pixels) or look for the corresponding mask file.
    
    line_identifier = PalmLineIdentifier()
    
    categories = ['MALE', 'FEMALE']
    
    # Clear output directory to ensure fresh start
    if os.path.exists(output_dir):
        print(f"Clearing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        
        if not os.path.exists(category_input_path):
            print(f"Warning: Directory {category_input_path} does not exist. Skipping.")
            continue
            
        print(f"Processing {category}...")
        
        # Walk through the directory to find all files ending with 'lines_enhanced.jpg'
        files = [f for f in os.listdir(category_input_path) if f.endswith('lines_enhanced.jpg')]
        
        for filename in tqdm(files, desc=f"Organizing {category}"):
            file_path = os.path.join(category_input_path, filename)
            
            # Read image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            # 1. Rotate 90 degrees clockwise
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # 2. Identify Hand Side (Left/Right)
            # We use a combination of heuristics:
            # A. Convexity Defects (from PalmLineIdentifier)
            # B. Center of Mass (COM) vs Bounding Box Center
            
            base_name = filename.replace('_lines_enhanced.jpg', '')
            bg_extracted_path = os.path.join(category_input_path, f"{base_name}_bg_extracted.jpg")
            
            hand_side = 'unknown'
            
            if os.path.exists(bg_extracted_path):
                bg_extracted = cv2.imread(bg_extracted_path)
                if bg_extracted is not None:
                    # Create binary mask
                    gray_bg = cv2.cvtColor(bg_extracted, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_bg, 1, 255, cv2.THRESH_BINARY)
                    
                    # Method A: Defect based (Existing)
                    side_defect = line_identifier.detect_hand_side(mask)
                    
                    # Method B: Center of Mass vs Bounding Box Center
                    # For Left Hand (Thumb Left), COM should be to the left of the main palm body center.
                    # However, if we consider the whole hand, the thumb pulls the COM to the left.
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        # Bounding box
                        x, y, w, h = cv2.boundingRect(mask)
                        bbox_center_x = x + w // 2
                        
                        # Heuristic:
                        # If COM is to the left of Bbox Center -> Left Hand (Thumb on Left)
                        # If COM is to the right of Bbox Center -> Right Hand (Thumb on Right)
                        if cx < bbox_center_x:
                            side_com = 'left'
                        else:
                            side_com = 'right'
                    else:
                        side_com = 'right' # Default
                    
                    # Consensus or Priority
                    # Convexity defects are more specific to the thumb-index gap if found correctly.
                    # But COM is more robust to noise.
                    # Let's trust COM for now as the user reported issues.
                    hand_side = side_com
            
            # Create output directory structure: output_dir/category/hand_side
            # e.g. output/final_dataset/MALE/left/
            target_dir = os.path.join(output_dir, category, hand_side)
            os.makedirs(target_dir, exist_ok=True)
            
            # Save the rotated image
            output_filename = f"{base_name}_processed.jpg"
            output_path = os.path.join(target_dir, output_filename)
            cv2.imwrite(output_path, rotated_image)

if __name__ == "__main__":
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Input: The output of the previous step
    input_directory = os.path.join(project_root, 'output', 'preprocessed_dataset')
    
    # Output: New folder for the final organized dataset
    output_directory = os.path.join(project_root, 'output', 'training_dataset')
    
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    
    organize_and_process_lines(input_directory, output_directory)
    print("Organization and processing complete!")
