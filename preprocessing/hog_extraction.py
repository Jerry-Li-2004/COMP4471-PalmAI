"""
HOG (Histogram of Oriented Gradients) Feature Extraction Module
Captures distribution of gradient orientations for palm structure analysis.
"""
import cv2
import numpy as np
from skimage.feature import hog
from typing import Tuple, Optional

class HOGExtractor:
    """Extract HOG features from images."""
    
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        visualize: bool = False
    ):
        """
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell (pixels)
            cells_per_block: Number of cells in each block
            visualize: Whether to generate visualization
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
    
    def extract(
        self,
        image: np.ndarray,
        return_visualization: bool = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract HOG features from grayscale image.
        
        Args:
            image: Input grayscale image
            return_visualization: Override class visualization setting
            
        Returns:
            Tuple of (hog_features, hog_image) if visualize=True
            Otherwise just hog_features
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        visualize = return_visualization if return_visualization is not None else self.visualize
        
        if visualize:
            features, hog_image = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=True,
                feature_vector=True
            )
            return features, hog_image
        else:
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                feature_vector=True
            )
            return features, None
    
    def extract_and_visualize(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract HOG features and create side-by-side visualization.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Tuple of (features, hog_image, combined_visualization)
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract with visualization
        features, hog_image = self.extract(gray, return_visualization=True)
        
        # Normalize HOG image for better visualization
        hog_image_rescaled = ((hog_image - hog_image.min()) / 
                              (hog_image.max() - hog_image.min()) * 255).astype(np.uint8)
        
        # Create side-by-side visualization
        # Resize images to same height if needed
        h1, w1 = gray.shape
        h2, w2 = hog_image_rescaled.shape
        
        if h1 != h2:
            scale = h1 / h2
            hog_image_rescaled = cv2.resize(
                hog_image_rescaled,
                (int(w2 * scale), h1)
            )
        
        # Convert grayscale to BGR for concatenation
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hog_bgr = cv2.cvtColor(hog_image_rescaled, cv2.COLOR_GRAY2BGR)
        
        # Concatenate
        combined = np.hstack([gray_bgr, hog_bgr])
        
        return features, hog_image_rescaled, combined
