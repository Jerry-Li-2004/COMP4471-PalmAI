import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_grayscale_range(image_path, lower_bound=(80, 80, 80), upper_bound=(100, 100, 100)):
    """
    Extract pixels within a specific grayscale range from an image.

    Args:
        image_path (str): Path to the input image
        lower_bound (tuple): Lower bound of grayscale range (R,G,B)
        upper_bound (tuple): Upper bound of grayscale range (R,G,B)

    Returns:
        tuple: (original_image, extracted_image, mask)
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert bounds to numpy arrays
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)

    # Create mask for pixels within the specified range
    mask = cv2.inRange(image_rgb, lower, upper)

    # Extract pixels using the mask
    extracted = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return image_rgb, extracted, mask


def display_results(original, extracted, mask):
    """Display the original image, extracted pixels, and mask."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(extracted)
    plt.title('Extracted Pixels')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_results(extracted, mask, extracted_path='extracted_pixels.png', mask_path='mask.png'):
    """Save the extracted image and mask to files."""
    # Convert extracted image back to BGR for saving
    extracted_bgr = cv2.cvtColor(extracted, cv2.COLOR_RGB2BGR)
    cv2.imwrite(extracted_path, extracted_bgr)
    cv2.imwrite(mask_path, mask)
    print(f"Extracted pixels saved as: {extracted_path}")
    print(f"Mask saved as: {mask_path}")


def main():
    # Get image path from user
    image_path = input("Enter the path to your image: ").strip()

    try:
        # Extract pixels in the specified range
        original, extracted, mask = extract_grayscale_range(image_path)

        # Display results
        display_results(original, extracted, mask)

        # Ask if user wants to save results
        save_choice = input(
            "Do you want to save the results? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results(extracted, mask)

        # Print some statistics
        total_pixels = mask.size
        selected_pixels = np.count_nonzero(mask)
        percentage = (selected_pixels / total_pixels) * 100

        print(f"\nImage Statistics:")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Pixels in range: {selected_pixels:,}")
        print(f"Percentage: {percentage:.2f}%")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check if the image path is correct and the image is valid.")

# Alternative function for custom range


def extract_custom_range():
    """Allow user to specify custom grayscale range."""
    print("Enter custom grayscale range:")
    try:
        r_low = int(input("Lower bound R value (0-255): "))
        g_low = int(input("Lower bound G value (0-255): "))
        b_low = int(input("Lower bound B value (0-255): "))

        r_high = int(input("Upper bound R value (0-255): "))
        g_high = int(input("Upper bound G value (0-255): "))
        b_high = int(input("Upper bound B value (0-255): "))

        lower_bound = (r_low, g_low, b_low)
        upper_bound = (r_high, g_high, b_high)

        return lower_bound, upper_bound

    except ValueError:
        print("Invalid input. Using default range (80,80,80) to (100,100,100)")
        return (30, 30, 30), (100, 100, 100)


if __name__ == "__main__":
    print("Grayscale Range Extractor")
    print("=" * 30)

    # Ask if user wants custom range
    custom = input("Use custom range? (y/n, default n): ").strip().lower()

    if custom == 'y':
        lower, upper = extract_custom_range()
        print(f"Using custom range: {lower} to {upper}")
        # You would modify the extract_grayscale_range function call to use these values
    else:
        print("Using default range: (80,80,80) to (100,100,100)")

    main()
