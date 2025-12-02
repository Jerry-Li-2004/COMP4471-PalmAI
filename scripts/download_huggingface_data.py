"""
Script to download the Open Palm Hand Images Dataset from Hugging Face.

Prerequisites:
1. Install huggingface_hub: pip install huggingface_hub
2. (Optional) Login to Hugging Face if dataset requires authentication:
   huggingface-cli login
"""

import os
from huggingface_hub import snapshot_download

def download_huggingface_dataset():
    """
    Downloads the Open Palm Hand Images Dataset from Hugging Face.
    """
    # Dataset identifier on Hugging Face
    dataset_id = "ud-biometrics/open-palm-hand-images"
    
    # Target directory
    target_dir = os.path.join(os.path.dirname(__file__), '../data/huggingface_palms')
    
    print("=" * 60)
    print("Hugging Face Dataset Downloader")
    print("Dataset: Open Palm Hand Images")
    print("=" * 60)
    print()
    print(f"Downloading dataset: {dataset_id}")
    print(f"Target directory: {target_dir}")
    print("-" * 60)
    
    try:
        # Download the dataset
        print("\nDownloading... (this may take a while)")
        
        downloaded_path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        print("\n✓ Dataset downloaded successfully!")
        print(f"Location: {downloaded_path}")
        
        # List contents
        print("\nDataset contents:")
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(target_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            if level == 0:
                print(f"{indent}📁 {os.path.basename(target_dir)}/")
            else:
                print(f"{indent}📁 {folder_name}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files per directory
                print(f"{subindent}📄 {file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
            
            if level > 2:  # Limit depth
                break
                
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n⚠️  Authentication required.")
            print("Please login to Hugging Face:")
            print("  huggingface-cli login")
        
        return False

if __name__ == "__main__":
    download_huggingface_dataset()
