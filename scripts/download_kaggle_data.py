"""
Script to download the Hands and Palm Images Dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Scroll to 'API' section
   - Click 'Create New Token' (downloads kaggle.json)
   - Place kaggle.json in ~/.kaggle/ directory
   - On Mac/Linux: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys

def download_kaggle_dataset():
    """
    Downloads the Hands and Palm Images Dataset from Kaggle.
    """
    # Dataset identifier on Kaggle
    dataset_id = "shyambhu/hands-and-palm-images-dataset"
    
    # Target directory
    target_dir = os.path.join(os.path.dirname(__file__), '../data/kaggle_hands')
    
    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_id}")
    print(f"Target directory: {target_dir}")
    print("-" * 50)
    
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
            
            # List contents
            print("\nDataset contents:")
            for item in os.listdir(target_dir):
                item_path = os.path.join(target_dir, item)
                if os.path.isdir(item_path):
                    num_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    print(f"  📁 {item}/ ({num_files} files)")
                else:
                    print(f"  📄 {item}")
        else:
            print("\n✗ Error downloading dataset:")
            print(result.stderr)
            
            if "401" in result.stderr or "Unauthorized" in result.stderr:
                print("\n⚠️  Authentication issue. Please ensure:")
                print("1. You have a Kaggle account")
                print("2. kaggle.json is in ~/.kaggle/")
                print("3. File permissions: chmod 600 ~/.kaggle/kaggle.json")
            
    except FileNotFoundError:
        print("\n✗ Error: 'kaggle' command not found")
        print("\nPlease install the Kaggle CLI:")
        print("  pip install kaggle")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Kaggle Dataset Downloader")
    print("Dataset: Hands and Palm Images")
    print("=" * 50)
    print()
    
    download_kaggle_dataset()
