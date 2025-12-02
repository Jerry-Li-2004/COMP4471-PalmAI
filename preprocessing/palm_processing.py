# import cv2
# import numpy as np
# import os
# import glob


# def extract_edges_simple(input_dir, output_dir, num_samples=5):
#     """Simple edge extraction for training dataset"""

#     os.makedirs(output_dir, exist_ok=True)

#     # Get image files
#     image_paths = glob.glob(os.path.join(input_dir, "*.*"))
#     image_paths = [p for p in image_paths if p.lower().endswith(
#         ('.jpg', '.jpeg', '.png', '.bmp'))]

#     # Use samples or create demo images
#     if not image_paths:
#         print("No images found. Creating demo images...")
#         for i in range(num_samples):
#             img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
#             cv2.imwrite(os.path.join(input_dir, f"demo_{i}.jpg"), img)
#         image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

#     image_paths = image_paths[:num_samples]

#     # Process each image
#     for i, img_path in enumerate(image_paths):
#         # Load and convert to grayscale
#         img = cv2.imread(img_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Blur and detect edges
#         blurred = cv2.GaussianBlur(gray, (5, 5), 1)
#         edges = cv2.Canny(blurred, 50, 150)

#         # Save results
#         sample_dir = os.path.join(output_dir, f"sample_{i+1}")
#         os.makedirs(sample_dir, exist_ok=True)

#         cv2.imwrite(os.path.join(sample_dir, "original.jpg"), img)
#         cv2.imwrite(os.path.join(sample_dir, "grayscale.jpg"), gray)
#         cv2.imwrite(os.path.join(sample_dir, "edges.jpg"), edges)

#         print(f"Processed sample {i+1}: {os.path.basename(img_path)}")

#     print(f"Edge extraction complete! Check {output_dir}")


# # Run the simple version
# if __name__ == "__main__":
#     extract_edges_simple("../output/training_dataset/MALE",
#                          "../output/training_dataset/EDGE", 5)

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import time


class EdgeExtractor:
    def __init__(self):
        self.original = None
        self.gray = None
        self.edges = None

    def load_image(self, image_path):
        """Load and convert image to grayscale"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        return self.gray

    def preprocess(self, blur_kernel=(5, 5), sigma=1.0):
        """Apply Gaussian blur to reduce noise"""
        if self.gray is None:
            raise ValueError("Load an image first")
        self.blurred = cv2.GaussianBlur(self.gray, blur_kernel, sigma)
        return self.blurred

    def extract_edges(self, method='canny', low_threshold=50, high_threshold=150):
        """Extract edges using Canny edge detection"""
        if self.gray is None:
            raise ValueError("Load and preprocess an image first")

        if method == 'canny':
            self.edges = cv2.Canny(self.blurred, low_threshold, high_threshold)
        else:
            raise ValueError("Only 'canny' method is supported")

        return self.edges

    def save_edge_only(self, output_dir, filename):
        """Save only the edge image"""
        os.makedirs(output_dir, exist_ok=True)

        # Save only edge image
        output_path = os.path.join(output_dir, f"{filename}_edge.jpg")
        cv2.imwrite(output_path, self.edges)
        return output_path


def process_all_images_edge_only(input_dir, output_dir):
    """
    Process ALL images in the input directory and extract ONLY edge images

    Args:
        input_dir: Directory containing training images
        output_dir: Directory to save edge images
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files from input directory
    image_extensions = ['*.jpg', '*.jpeg',
                        '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []

    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, extension)))
        image_paths.extend(
            glob.glob(os.path.join(input_dir, extension.upper())))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return 0

    print(f"Found {len(image_paths)} images in {input_dir}")
    print("Starting edge extraction...")

    # Process each image
    extractor = EdgeExtractor()
    successful_processing = 0
    failed_images = []

    start_time = time.time()

    for i, image_path in enumerate(image_paths):
        try:
            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                print(
                    f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

            # Load and process image
            extractor.load_image(image_path)
            extractor.preprocess()
            extractor.extract_edges()

            # Create filename
            filename = Path(image_path).stem

            # Save ONLY the edge image
            extractor.save_edge_only(output_dir, filename)
            successful_processing += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            failed_images.append((os.path.basename(image_path), str(e)))

    end_time = time.time()
    processing_time = end_time - start_time

    # Print summary
    print("\n" + "="*60)
    print("EDGE EXTRACTION COMPLETE - ALL IMAGES PROCESSED")
    print("="*60)
    print(f"Total images found: {len(image_paths)}")
    print(f"Successfully processed: {successful_processing}")
    print(f"Failed to process: {len(failed_images)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(
        f"Average time per image: {processing_time/len(image_paths):.2f} seconds")

    if failed_images:
        print(f"\nFailed images:")
        for img, error in failed_images[:10]:  # Show first 10 failures
            print(f"  - {img}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    return successful_processing


def create_progress_visualization(output_dir, max_images=20):
    """Create a visualization of sample edge images"""
    import matplotlib.pyplot as plt

    # Get all edge images
    edge_images = glob.glob(os.path.join(output_dir, "*_edge.jpg"))

    if not edge_images:
        print("No edge images found to visualize")
        return

    # Limit to max_images for visualization
    edge_images = edge_images[:max_images]

    # Create a grid of edge images
    num_images = len(edge_images)
    cols = min(5, num_images)  # 5 columns max
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    fig.suptitle(
        f'Edge Detection Results - Sample of {num_images} images', fontsize=16, y=1.02)

    if num_images == 1:
        axes = np.array([axes])
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, edge_path in enumerate(edge_images):
        row = i // cols
        col = i % cols

        try:
            # Read edge image
            edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

            # Plot edge image
            axes[row, col].imshow(edge_img, cmap='gray')
            axes[row, col].set_title(f'{Path(edge_path).stem}', fontsize=8)
            axes[row, col].axis('off')
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error\nloading image',
                                ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')

    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_detection_sample_grid.jpg'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def check_image_sizes(output_dir):
    """Check and report on the sizes of generated edge images"""
    edge_images = glob.glob(os.path.join(output_dir, "*_edge.jpg"))

    if not edge_images:
        return

    sizes = []
    for edge_path in edge_images:
        img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sizes.append(img.shape)

    if sizes:
        heights = [size[0] for size in sizes]
        widths = [size[1] for size in sizes]

        print(f"\nImage Size Analysis:")
        print(f"Total edge images: {len(sizes)}")
        print(f"Height range: {min(heights)} - {max(heights)} pixels")
        print(f"Width range: {min(widths)} - {max(widths)} pixels")
        print(f"Most common size: {max(set(sizes), key=sizes.count)}")


# Fast batch processing version
def batch_process_edges(input_dir, output_dir):
    """
    Faster batch processing for large datasets
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg',
                        '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []

    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, extension)))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return 0

    print(f"Batch processing {len(image_paths)} images...")

    successful = 0
    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        try:
            # Fast processing without class overhead
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1)
            edges = cv2.Canny(blurred, 50, 150)

            filename = Path(img_path).stem
            output_path = os.path.join(output_dir, f"{filename}_edge.jpg")
            cv2.imwrite(output_path, edges)
            successful += 1

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images...")

        except Exception as e:
            print(f"Error with {Path(img_path).name}: {e}")

    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f} seconds")
    return successful


# Main execution
if __name__ == "__main__":
    # Configuration
    # Directory containing all 400 images
    INPUT_DIR = "../output/training_dataset/MALE"
    # Output directory for edge images
    OUTPUT_DIR = "../output/training_dataset/EDGE_MALE"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting edge extraction for ALL images...")
    print("="*50)

    # Option 1: Use detailed processing with progress tracking
    processed_count = process_all_images_edge_only(INPUT_DIR, OUTPUT_DIR)

    # Option 2: Use faster batch processing (uncomment if you prefer speed)
    # processed_count = batch_process_edges(INPUT_DIR, OUTPUT_DIR)

    # Create visualization of results
    if processed_count > 0:
        print("\nCreating visualization of sample results...")
        create_progress_visualization(OUTPUT_DIR, max_images=20)

        # Check image sizes
        check_image_sizes(OUTPUT_DIR)

        # Final summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        edge_files = glob.glob(os.path.join(OUTPUT_DIR, "*_edge.jpg"))
        print(f"Edge images generated: {len(edge_files)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("All images have been processed successfully!")
    else:
        print("No images were processed. Please check the input directory path.")
