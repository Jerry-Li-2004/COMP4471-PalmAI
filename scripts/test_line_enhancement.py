import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.line_identification import PalmLineIdentifier

def ridge_detection(img, sigma=1.0):
    """
    Detect ridges (lines) using Hessian matrix eigenvalues.
    """
    # Ensure float
    img = img.astype(float)
    
    # Gaussian smoothing
    img_smooth = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Second derivatives
    dx = cv2.Sobel(img_smooth, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize=3)
    
    # Eigenvalues of Hessian
    # H = [[dxx, dxy], [dxy, dyy]]
    # lambda = ((dxx + dyy) +/- sqrt((dxx - dyy)^2 + 4*dxy^2)) / 2
    
    tmp = np.sqrt((dxx - dyy)**2 + 4*dxy**2)
    l1 = (dxx + dyy + tmp) / 2
    l2 = (dxx + dyy - tmp) / 2
    
    # For dark lines on bright background, we want large positive curvature in one direction
    # We are looking for valleys.
    # The eigenvalues represent curvature.
    # We want one large positive eigenvalue (curvature across the line) and one small one (curvature along the line).
    
    # Normalize to 0-255
    # We visualize the maximum curvature
    ridges = np.maximum(l1, l2)
    ridges[ridges < 0] = 0 # Only keep valleys (dark lines)
    
    return ridges

def test_enhancement():
    image_path = 'data/MALE/IMG_0001.JPG'
    output_dir = 'output/line_enhancement_test'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Read and resize
    image = cv2.imread(image_path)
    max_dim = 512
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, '00_original_gray.jpg'), gray)
    
    # Method 1: Current Canny (via PalmLineIdentifier logic)
    print("Testing Method 1: Canny...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    canny = cv2.Canny(blurred, lower, upper)
    cv2.imwrite(os.path.join(output_dir, '01_canny_default.jpg'), canny)
    
    # Method 2: Aggressive CLAHE + Canny
    print("Testing Method 2: Aggressive CLAHE + Canny...")
    clahe_agg = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced_agg = clahe_agg.apply(gray)
    blurred_agg = cv2.GaussianBlur(enhanced_agg, (5, 5), 0)
    canny_agg = cv2.Canny(blurred_agg, 30, 100) # Fixed loose thresholds
    cv2.imwrite(os.path.join(output_dir, '02_canny_aggressive.jpg'), canny_agg)
    
    # Method 3: Sobel Gradient Magnitude
    print("Testing Method 3: Sobel Gradient...")
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Threshold to clean up
    _, mag_thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, '03_sobel_magnitude.jpg'), magnitude)
    cv2.imwrite(os.path.join(output_dir, '03_sobel_thresh.jpg'), mag_thresh)
    
    # Method 4: Ridge Detection (Hessian)
    print("Testing Method 4: Ridge Detection...")
    ridges = ridge_detection(gray, sigma=1.0)
    ridges_norm = cv2.normalize(ridges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Invert because ridges returns high values for lines
    cv2.imwrite(os.path.join(output_dir, '04_ridges.jpg'), ridges_norm)
    
    # Method 5: Top-Hat Transform (Morphology)
    # Highlights bright objects on dark background. 
    # For dark lines on light background, use Black-Hat.
    print("Testing Method 5: Black-Hat Morphology...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Enhance contrast of blackhat
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bh_thresh = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, '05_blackhat.jpg'), blackhat)
    cv2.imwrite(os.path.join(output_dir, '05_blackhat_thresh.jpg'), bh_thresh)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    test_enhancement()
