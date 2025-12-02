"""
Filter 11k Hands Dataset to only include palmar images with clearly visible palm lines.
Uses edge detection and line detection to verify palm line visibility.
"""
import pandas as pd
import os
import shutil
import cv2
import numpy as np
from pathlib import Path

def detect_palm_lines(image_path: str, debug: bool = False) -> dict:
    """
    Detect if palm lines are clearly visible in the image.
    
    Args:
        image_path: Path to the image
        debug: If True, save debug visualization
        
    Returns:
        Dictionary with detection results:
            - 'has_lines': Boolean indicating if lines are detected
            - 'line_count': Number of detected lines
            - 'line_strength': Average strength of detected lines
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {'has_lines': False, 'line_count': 0, 'line_strength': 0}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=20
    )
    
    # Calculate metrics
    line_count = 0 if lines is None else len(lines)
    
    # Calculate line strength (average length of detected lines)
    line_strength = 0
    if lines is not None and len(lines) > 0:
        lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            lengths.append(length)
        line_strength = np.mean(lengths)
    
    # Determine if palm lines are visible
    # Criteria: At least 10 lines detected with reasonable strength
    has_lines = line_count >= 10 and line_strength > 50
    
    return {
        'has_lines': has_lines,
        'line_count': line_count,
        'line_strength': line_strength
    }

def filter_palmar_images_with_lines(
    data_dir: str,
    csv_path: str,
    output_dir: str,
    copy_files: bool = True,
    check_lines: bool = True,
    sample_check: int = None
):
    """
    Filter dataset to only include palmar images with visible palm lines.
    
    Args:
        data_dir: Root directory containing Hands/Hands/ images
        csv_path: Path to HandInfo.csv
        output_dir: Directory to save filtered images
        copy_files: If True, copy files; if False, create symlinks
        check_lines: If True, verify palm lines are visible
        sample_check: If set, only check first N images (for testing)
    """
    print("=" * 80)
    print("11k Hands Dataset - Palmar Image Filter with Line Detection")
    print("=" * 80)
    
    # Load metadata
    print(f"\nLoading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Clean string columns
    str_cols = df.select_dtypes(include=['object']).columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
    
    print(f"Total images in dataset: {len(df)}")
    
    # Filter for palmar images
    print("\nStep 1: Filtering for palmar images...")
    palmar_df = df[df['aspectOfHand'].str.contains('palmar', case=False)]
    
    print(f"  Palmar images found: {len(palmar_df)}")
    print(f"  Dorsal images (removed): {len(df) - len(palmar_df)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Source directory
    source_dir = os.path.join(data_dir, 'Hands', 'Hands')
    
    if not os.path.exists(source_dir):
        print(f"\n❌ Error: Source directory not found: {source_dir}")
        return
    
    # Check for palm lines if requested
    if check_lines:
        print(f"\nStep 2: Checking for visible palm lines...")
        
        valid_images = []
        line_detection_results = []
        
        images_to_check = palmar_df.head(sample_check) if sample_check else palmar_df
        
        for idx, row in images_to_check.iterrows():
            img_name = row['imageName']
            source_path = os.path.join(source_dir, img_name)
            
            if not os.path.exists(source_path):
                continue
            
            # Detect palm lines
            result = detect_palm_lines(source_path)
            result['imageName'] = img_name
            line_detection_results.append(result)
            
            if result['has_lines']:
                valid_images.append(idx)
            
            # Progress update
            if len(line_detection_results) % 100 == 0:
                valid_count = sum(1 for r in line_detection_results if r['has_lines'])
                print(f"  Checked {len(line_detection_results)}/{len(images_to_check)} images... "
                      f"({valid_count} with visible lines)")
        
        # Filter dataframe to only valid images
        palmar_df = palmar_df.loc[valid_images]
        
        valid_count = len(palmar_df)
        total_checked = len(line_detection_results)
        print(f"\n  ✓ Images with visible palm lines: {valid_count}/{total_checked} "
              f"({valid_count/total_checked*100:.1f}%)")
        
        # Save line detection results
        results_df = pd.DataFrame(line_detection_results)
        results_path = os.path.join(output_dir, 'line_detection_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"  ✓ Line detection results saved to: {results_path}")
    
    # Copy/link filtered images
    print(f"\nStep 3: {'Copying' if copy_files else 'Linking'} filtered images...")
    
    copied_count = 0
    missing_count = 0
    
    for idx, row in palmar_df.iterrows():
        img_name = row['imageName']
        source_path = os.path.join(source_dir, img_name)
        dest_path = os.path.join(output_dir, img_name)
        
        if not os.path.exists(source_path):
            missing_count += 1
            continue
        
        try:
            if copy_files:
                shutil.copy2(source_path, dest_path)
            else:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.symlink(source_path, dest_path)
            
            copied_count += 1
            
            if copied_count % 500 == 0:
                print(f"  Processed {copied_count}/{len(palmar_df)} images...")
                
        except Exception as e:
            print(f"  Error processing {img_name}: {e}")
    
    # Save filtered CSV
    filtered_csv_path = os.path.join(output_dir, 'HandInfo_palmar_with_lines.csv')
    palmar_df.to_csv(filtered_csv_path, index=False)
    
    print(f"\n✓ Successfully processed {copied_count} images")
    print(f"✓ Filtered CSV saved to: {filtered_csv_path}")
    
    if missing_count > 0:
        print(f"⚠️  Warning: {missing_count} images listed in CSV were not found")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Filtering Statistics:")
    print("=" * 80)
    print(f"Original dataset size: {len(df)} images")
    print(f"After palmar filter: {len(palmar_df)} images")
    if check_lines:
        print(f"After line detection filter: {copied_count} images")
        print(f"Rejection rate: {(1 - copied_count/len(df))*100:.1f}%")
    print("=" * 80)
    
    return palmar_df

def main():
    """Main function to filter the dataset."""
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '../data/11k_hands')
    csv_path = os.path.join(data_dir, 'HandInfo.csv')
    output_dir = os.path.join(data_dir, 'palmar_with_lines')
    
    # Check if data exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: HandInfo.csv not found at {csv_path}")
        print("Please ensure the 11k Hands dataset is downloaded.")
        return
    
    # Filter dataset
    print("\nConfiguration:")
    print("  - Background removal: No")
    print("  - Line detection: Yes (Canny + Hough)")
    print("  - Minimum lines: 10")
    print("  - Minimum line strength: 50 pixels")
    print()
    
    palmar_df = filter_palmar_images_with_lines(
        data_dir=data_dir,
        csv_path=csv_path,
        output_dir=output_dir,
        copy_files=True,
        check_lines=True,
        sample_check=None  # Set to a number (e.g., 100) for testing
    )
    
    # Show sample of filtered data
    if palmar_df is not None and len(palmar_df) > 0:
        print("\nSample of filtered data:")
        print(palmar_df[['id', 'age', 'gender', 'aspectOfHand', 'imageName']].head(10))

if __name__ == "__main__":
    main()
