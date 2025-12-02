from roboflow import Roboflow
import os
import sys

def download_dataset(api_key):
    """
    Downloads the 'palm-lines-recognition' dataset from Roboflow.
    """
    try:
        rf = Roboflow(api_key=api_key)
        print("Authenticating...")
        # Using the 'dstu' workspace and 'palm-lines-recognition' project
        project = rf.workspace("dstu").project("palm-lines-recognition")
        
        # Try downloading the latest version
        # We'll try a few common formats if one fails
        formats_to_try = ["yolov8", "yolov5pytorch", "coco"]
        
        dataset = None
        for fmt in formats_to_try:
            try:
                print(f"Attempting download with format: {fmt}...")
                # We use version 1 as a baseline, but ideally we'd list versions.
                # Roboflow SDK usually picks the latest if we don't specify, but let's be explicit or try/catch.
                # project.version(1) is standard.
                dataset = project.version(1).download(fmt)
                break
            except Exception as e:
                print(f"Failed to download with format {fmt}: {e}")
                if "zip" in str(e).lower():
                    print("This usually means the export format is not available or the export failed on the server.")
        
        if dataset:
            print(f"Dataset downloaded to: {dataset.location}")
            
            # Create a symlink or move logic if needed, but Roboflow usually downloads to a folder
            # Let's ensure our data loader knows where to look.
            # We'll print the path clearly.
            
            target_dir = os.path.join(os.path.dirname(__file__), '../data/roboflow')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            print(f"\nSUCCESS! Data is at: {dataset.location}")
            print(f"You can now link this to {target_dir} or update the loader path.")
            
        else:
            print("\nCould not download dataset in any attempted format.")
            print("Please check the project page: https://universe.roboflow.com/dstu/palm-lines-recognition")
        
    except Exception as e:
        print(f"\nFatal Error: {e}")
        print("Please ensure your API key is correct.")
        print("To get your key: https://app.roboflow.com/ -> Settings -> Roboflow Keys")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your Roboflow API Key: ")
        
    if api_key:
        download_dataset(api_key)
    else:
        print("API Key is required.")
