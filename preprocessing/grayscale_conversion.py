"""
Grayscale Conversion Module
Transforms color images to grayscale for simplified processing.
"""
import cv2
import numpy as np

class GrayscaleConverter:
    """Convert images to grayscale."""
    
    def __init__(self, method='weighted'):
        """
        Args:
            method: 'weighted' (standard CV), 'average', or 'luminosity'
        """
        self.method = method
    
    def convert_weighted(self, image: np.ndarray) -> np.ndarray:
        """
        Standard OpenCV grayscale conversion (weighted RGB).
        Uses formula: 0.299*R + 0.587*G + 0.114*B
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def convert_average(self, image: np.ndarray) -> np.ndarray:
        """
        Simple average of RGB channels.
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        return np.mean(image, axis=2).astype(np.uint8)
    
    def convert_luminosity(self, image: np.ndarray) -> np.ndarray:
        """
        Luminosity method (similar to weighted but different coefficients).
        Uses formula: 0.21*R + 0.72*G + 0.07*B
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        # BGR to RGB, then apply luminosity formula
        b, g, r = cv2.split(image)
        gray = 0.07 * b + 0.72 * g + 0.21 * r
        return gray.astype(np.uint8)
    
    def convert(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale using configured method.
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        if self.method == 'weighted':
            return self.convert_weighted(image)
        elif self.method == 'average':
            return self.convert_average(image)
        elif self.method == 'luminosity':
            return self.convert_luminosity(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
