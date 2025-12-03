import os
from PIL import Image
import glob
import shutil


def find_largest_image(folder_path):
    """Find the largest image dimensions in the folder"""
    max_width = 0
    max_height = 0
    largest_image_name = ""

    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png',
                        '*.bmp', '*.gif', '*.tiff', '*.webp']

    for extension in image_extensions:
        for image_path in glob.glob(os.path.join(folder_path, extension)):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width > max_width or height > max_height:
                        max_width = max(max_width, width)
                        max_height = max(max_height, height)
                        largest_image_name = os.path.basename(image_path)
                    print(
                        f"Scanned: {os.path.basename(image_path)} - Size: {width}x{height}")
            except Exception as e:
                print(f"Error reading {image_path}: {e}")

    return max_width, max_height, largest_image_name


def resize_images_to_target(folder_path, target_width, target_height, output_folder):
    """Resize all images to target size, filling smaller images with black, keep original names"""

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    image_extensions = ['*.jpg', '*.jpeg', '*.png',
                        '*.bmp', '*.gif', '*.tiff', '*.webp']
    processed_count = 0
    success_count = 0
    error_count = 0

    for extension in image_extensions:
        for image_path in glob.glob(os.path.join(folder_path, extension)):
            try:
                processed_count += 1
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_folder, filename)

                # Skip if file already exists in output folder (avoid reprocessing)
                if os.path.exists(output_path):
                    print(f"Skipped (already exists): {filename}")
                    success_count += 1
                    continue

                with Image.open(image_path) as img:
                    original_mode = img.mode

                    # Convert to RGB if necessary (for PNG with transparency)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create a black background for transparent images
                        background = Image.new('RGB', img.size, (0, 0, 0))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()
                                         [-1] if img.mode == 'RGBA' else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Get current dimensions
                    current_width, current_height = img.size

                    # Create new image with black background
                    new_img = Image.new(
                        'RGB', (target_width, target_height), (0, 0, 0))

                    # Calculate position to paste the original image (centered)
                    x_offset = (target_width - current_width) // 2
                    y_offset = (target_height - current_height) // 2

                    # Paste the original image onto the black background
                    new_img.paste(img, (x_offset, y_offset))

                    # Determine output format based on original extension
                    file_extension = os.path.splitext(filename)[1].lower()
                    if file_extension in ['.jpg', '.jpeg']:
                        format_type = 'JPEG'
                    elif file_extension == '.png':
                        format_type = 'PNG'
                    elif file_extension == '.bmp':
                        format_type = 'BMP'
                    elif file_extension == '.gif':
                        format_type = 'GIF'
                    elif file_extension == '.tiff':
                        format_type = 'TIFF'
                    elif file_extension == '.webp':
                        format_type = 'WEBP'
                    else:
                        format_type = 'JPEG'  # default

                    # Save with original filename and appropriate format
                    new_img.save(output_path, format=format_type)

                    success_count += 1
                    print(
                        f"✅ Resized: {filename} ({current_width}x{current_height} -> {target_width}x{target_height})")

            except Exception as e:
                error_count += 1
                print(
                    f"❌ Error processing {os.path.basename(image_path)}: {e}")

    return processed_count, success_count, error_count


def create_output_folder_structure(input_folder):
    """Create a standardized output folder structure"""
    base_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = os.path.join(input_folder, f"{base_name}_resized")

    # If folder exists, add number suffix
    counter = 1
    original_output_folder = output_folder
    while os.path.exists(output_folder):
        output_folder = f"{original_output_folder}_{counter}"
        counter += 1

    return output_folder


def main():
    # Specify your folder path here
    folder_path = input(
        "Enter the path to the folder containing images: ").strip()

    # Remove quotes if user pasted path with quotes
    folder_path = folder_path.strip('"\'')

    if not os.path.exists(folder_path):
        print("Error: The specified folder does not exist!")
        return

    print("Finding the largest image in the folder...")
    max_width, max_height, largest_image = find_largest_image(folder_path)

    if max_width == 0 or max_height == 0:
        print("No valid images found in the folder!")
        return

    print(f"\n📊 Summary:")
    print(f"Largest image: {largest_image}")
    print(f"Target size: {max_width}x{max_height}")

    # Create output folder
    output_folder = create_output_folder_structure(folder_path)

    print(f"\n🔄 Resizing all images to {max_width}x{max_height}...")
    print(f"Output folder: {output_folder}")

    confirm = input("Proceed with resizing? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    processed_count, success_count, error_count = resize_images_to_target(
        folder_path, max_width, max_height, output_folder
    )

    print(f"\n🎉 Resizing completed!")
    print(f"📁 Output folder: {output_folder}")
    print(f"📊 Statistics:")
    print(f"   Total images found: {processed_count}")
    print(f"   Successfully resized: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Target size: {max_width}x{max_height}")

    # Show sample of processed files
    print(f"\n📋 All images maintain their original filenames")
    print(f"   Example: image.jpg -> {output_folder}/image.jpg")


if __name__ == "__main__":
    main()
