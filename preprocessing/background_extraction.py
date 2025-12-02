"""
Background Extraction Module
Isolates the palm/hand from the surrounding background using GrabCut algorithm.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class BackgroundExtractor:
    """Extract foreground (hand/palm) from background."""
    
    def __init__(self, method='grabcut'):
        """
        Args:
            method: 'grabcut' or 'threshold' for background removal
        """
        self.method = method
    
    def extract_grabcut(
        self, 
        image: np.ndarray,
        iterations: int = 5,
        margin: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract foreground using GrabCut algorithm.
        
        Args:
            image: Input BGR image
            iterations: Number of GrabCut iterations
            margin: Margin from image edges for initial rectangle
            
        Returns:
            Tuple of (foreground_image, mask)
        """
        # Create initial mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Background and foreground models (required by GrabCut)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle around the hand (assuming hand is centered)
        h, w = image.shape[:2]
        rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        # Apply GrabCut
        cv2.grabCut(
            image, 
            mask, 
            rect, 
            bgd_model, 
            fgd_model, 
            iterations, 
            cv2.GC_INIT_WITH_RECT
        )
        
        # Create binary mask (0 and 2 are background, 1 and 3 are foreground)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        foreground = image * mask2[:, :, np.newaxis]
        
        return foreground, mask2
    
    def extract_threshold(
        self,
        image: np.ndarray,
        threshold: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple threshold-based background extraction.
        Assumes background is darker than hand.
        
        Args:
            image: Input BGR image
            threshold: Threshold value for binarization
            
        Returns:
            Tuple of (foreground_image, mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Normalize mask to 0-1
        mask = (mask / 255).astype('uint8')
        
        # Apply mask
        foreground = image * mask[:, :, np.newaxis]
        
        return foreground, mask
    
    def extract(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract foreground using configured method.
        
        Args:
            image: Input BGR image
            **kwargs: Additional arguments for specific method
            
        Returns:
            Tuple of (foreground_image, mask)
        """
        if self.method == 'grabcut':
            return self.extract_grabcut(image, **kwargs)
        elif self.method == 'threshold':
            return self.extract_threshold(image, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def visualize(
        self,
        original: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Create visualization showing original, mask, and result.
        
        Args:
            original: Original image
            foreground: Extracted foreground
            mask: Binary mask
            
        Returns:
            Concatenated visualization image
        """
        # Convert mask to 3-channel for visualization
        mask_vis = cv2.cvtColor((mask * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        
        # Concatenate horizontally
        vis = np.hstack([original, mask_vis, foreground])
        
        return vis
