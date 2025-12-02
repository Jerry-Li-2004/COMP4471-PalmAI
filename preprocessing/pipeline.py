"""
Integrated Preprocessing Pipeline
Combines background extraction, grayscale conversion, and HOG feature extraction.
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from background_extraction import BackgroundExtractor
from grayscale_conversion import GrayscaleConverter
from hog_extraction import HOGExtractor
from line_enhancement import LineEnhancer

class PreprocessingPipeline:
    """Complete preprocessing pipeline for palm images."""
    
    def __init__(
        self,
        background_method: str = 'grabcut',
        grayscale_method: str = 'weighted',
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (8, 8),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
        enhance_lines: bool = True
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            background_method: Method for background extraction
            grayscale_method: Method for grayscale conversion
            hog_orientations: Number of HOG orientation bins
            hog_pixels_per_cell: HOG cell size
            hog_cells_per_block: HOG block size
            enhance_lines: Whether to perform line enhancement
        """
        self.bg_extractor = BackgroundExtractor(method=background_method)
        self.gray_converter = GrayscaleConverter(method=grayscale_method)
        self.hog_extractor = HOGExtractor(
            orientations=hog_orientations,
            pixels_per_cell=hog_pixels_per_cell,
            cells_per_block=hog_cells_per_block,
            visualize=True
        )
        self.line_enhancer = LineEnhancer() if enhance_lines else None
    
    def process(
        self,
        image: np.ndarray,
        return_intermediates: bool = False
    ) -> Dict:
        """
        Process image through complete pipeline.
        
        Args:
            image: Input BGR image
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - 'hog_features': HOG feature vector
                - 'final_image': Final processed image (grayscale)
                - 'intermediates': Dict of intermediate results (if requested)
        """
        results = {}
        intermediates = {}
        
        # Step 1: Background Extraction
        foreground, bg_mask = self.bg_extractor.extract(image)
        intermediates['foreground'] = foreground
        intermediates['background_mask'] = bg_mask
        
        # Step 2: Grayscale Conversion
        gray = self.gray_converter.convert(foreground)
        intermediates['grayscale'] = gray
        
        # Step 3: Line Enhancement (Optional but recommended for training data)
        if self.line_enhancer:
            enhanced_lines = self.line_enhancer.enhance(gray)
            # Mask out background noise using the background mask
            if bg_mask is not None:
                enhanced_lines = cv2.bitwise_and(enhanced_lines, enhanced_lines, mask=bg_mask)
            intermediates['enhanced_lines'] = enhanced_lines
            feature_input = enhanced_lines
        else:
            feature_input = gray
        
        # Step 4: HOG Feature Extraction
        hog_features, hog_image = self.hog_extractor.extract(
            feature_input,
            return_visualization=True
        )
        intermediates['hog_image'] = hog_image
        
        # Prepare results
        results['hog_features'] = hog_features
        results['final_image'] = gray
        
        if return_intermediates:
            results['intermediates'] = intermediates
        
        return results
    
    def visualize_pipeline(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of all preprocessing steps.
        
        Args:
            image: Input BGR image
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image showing all steps
        """
        # Process with intermediates
        results = self.process(image, return_intermediates=True)
        intermediates = results['intermediates']
        
        # Prepare images for visualization
        original = image
        foreground = intermediates['foreground']
        gray = intermediates['grayscale']
        hog_image = intermediates['hog_image']
        
        # Normalize HOG image
        hog_vis = ((hog_image - hog_image.min()) / 
                   (hog_image.max() - hog_image.min()) * 255).astype(np.uint8)
        
        # Convert all to BGR for consistent display
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hog_bgr = cv2.cvtColor(hog_vis, cv2.COLOR_GRAY2BGR)
        
        # Resize all to same height
        h = original.shape[0]
        foreground = cv2.resize(foreground, (int(foreground.shape[1] * h / foreground.shape[0]), h))
        gray_bgr = cv2.resize(gray_bgr, (int(gray_bgr.shape[1] * h / gray_bgr.shape[0]), h))
        hog_bgr = cv2.resize(hog_bgr, (int(hog_bgr.shape[1] * h / hog_bgr.shape[0]), h))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original, '1. Original', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(foreground, '2. Background Removed', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(gray_bgr, '3. Grayscale', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(hog_bgr, '4. HOG Features', (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Concatenate horizontally
        vis = np.hstack([original, foreground, gray_bgr, hog_bgr])
        
        if save_path:
            cv2.imwrite(save_path, vis)
            print(f"Visualization saved to: {save_path}")
        
        return vis
