"""
Script to download the 11k Hands Dataset from Kaggle.
This dataset contains 11,076 hand images from 190 subjects.

Prerequisites:
1. Kaggle API credentials already set up in ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys

def download_11k_hands_dataset():
    """
    Downloads the 11k Hands Dataset from Kaggle.
    """
    # Dataset identifier on Kaggle
    dataset_id = "shyambhu/hands-and-palm-images-dataset"
    
    # Target directory
    target_dir = os.path.join(os.path.dirname(__file__), '../data/11k_hands')
    
    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print("=" * 60)
    print("11k Hands Dataset Downloader")
    print("Dataset: 11,076 hand images (190 subjects)")
    print("=" * 60)
    print()
    print(f"Downloading dataset: {dataset_id}")
    print(f"Target directory: {target_dir}")
    print("-" * 60)
    
    try:
        # Download using kaggle CLI
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_id,
            "-p", target_dir,
            "--unzip"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✓ Dataset downloaded successfully!")
            print(f"Location: {target_dir}")
            
            # Count images
            import glob
            jpg_files = glob.glob(os.path.join(target_dir, "**/*.jpg"), recursive=True)
            png_files = glob.glob(os.path.join(target_dir, "**/*.png"), recursive=True)
            total_images = len(jpg_files) + len(png_files)
            
            print(f"\n📊 Total images downloaded: {total_images}")
            
            # List directory structure
            print("\nDataset contents:")
            for item in os.listdir(target_dir):
                item_path = os.path.join(target_dir, item)
                if os.path.isdir(item_path):
                    num_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    print(f"  📁 {item}/ ({num_files} files)")
                else:
                    print(f"  📄 {item}")
                    
            return True
        else:
            print("\n✗ Error downloading dataset:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("\n✗ Error: 'kaggle' command not found")
        print("Please ensure Kaggle CLI is installed and configured.")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    download_11k_hands_dataset()
