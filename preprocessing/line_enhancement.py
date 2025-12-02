"""
Line Enhancement Module
Enhances palm lines using morphological operations (Black-Hat transform).
"""
import cv2
import numpy as np

class LineEnhancer:
    """Enhance visibility of palm lines."""
    
    def __init__(self, kernel_size: int = 15):
        """
        Args:
            kernel_size: Size of the structuring element for morphology.
                         Should be roughly the width of the palm lines.
        """
        self.kernel_size = kernel_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance lines in the image.
        
        Args:
            image: Input grayscale or BGR image.
            
        Returns:
            Enhanced image (grayscale) where lines are bright.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Black-Hat transform
        # This extracts dark regions (lines) from bright background (skin)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        
        # Normalize to full dynamic range
        enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Optional: Threshold to remove very faint noise
        # _, enhanced = cv2.threshold(enhanced, 20, 255, cv2.THRESH_TOZERO)
        
        return enhanced
