"""
Palm Line Identification Module
Identifies major palm lines (Heart, Head, Life, Fate) using edge detection and ROI heuristics.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

class PalmLineIdentifier:
    """Identify palm lines from preprocessed hand images."""
    
    def __init__(self):
        pass
        
    def detect_hand_side(self, mask: np.ndarray) -> str:
        """
        Detect if hand is left or right based on thumb position.
        Assumes palm is facing the camera and upright.
        
        Args:
            mask: Binary mask of the hand
            
        Returns:
            'left' or 'right'
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 'right' # Default
            
        cnt = max(contours, key=cv2.contourArea)
        
        # Find centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return 'right'
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Find convex hull and defects to locate thumb
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) <= 3:
            return 'right'
            
        try:
            defects = cv2.convexityDefects(cnt, hull)
        except:
            return 'right'
            
        if defects is None:
            return 'right'
            
        # Analyze defects to find the one between thumb and index
        # Usually the deepest defect
        max_depth = 0
        defect_pt = None
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > max_depth:
                max_depth = d
                defect_pt = tuple(cnt[f][0])
        
        if defect_pt:
            # If defect is to the left of centroid, thumb is likely on left -> Left Hand
            # If defect is to the right of centroid, thumb is likely on right -> Right Hand
            # Note: This is a heuristic and assumes upright orientation
            if defect_pt[0] < cx:
                return 'left'
            else:
                return 'right'
                
        return 'right'

    def get_palm_center(self, mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Find the center and radius of the largest inscribed circle in the hand mask.
        This corresponds to the palm center.
        """
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        return max_loc, max_val

    def extract_lines(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        side: str = 'auto'
    ) -> Dict[str, np.ndarray]:
        """
        Extract the four major palm lines.
        
        Args:
            image: Grayscale input image of the hand
            mask: Optional binary mask of the hand
            side: 'left', 'right', or 'auto'
            
        Returns:
            Dictionary with keys 'heart', 'head', 'life', 'fate' containing binary masks of the lines
        """
        if mask is None:
            _, mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            
        if side == 'auto':
            side = self.detect_hand_side(mask)
            
        # 1. Find Palm Center and Radius
        center, radius = self.get_palm_center(mask)
        cx, cy = center
        r = int(radius)
        
        # 2. Preprocessing for Line Detection
        # Enhance contrast in the palm area using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Canny Edge Detection with adjusted thresholds
        # Use a tighter range to capture lines but avoid skin texture
        v = np.median(image[mask > 0])
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)
        
        # 3. Create a Palm Mask to exclude fingers and boundaries
        # We focus on the area around the palm center
        palm_mask = np.zeros_like(mask)
        
        # Draw a circle for the main palm area, slightly larger than the inscribed circle
        # but eroded to avoid boundaries
        cv2.circle(palm_mask, (cx, cy), int(r * 1.6), 255, -1)
        
        # Intersect with original mask eroded to remove boundary artifacts
        eroded_hand_mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=2)
        palm_mask = cv2.bitwise_and(palm_mask, eroded_hand_mask)
        
        # Apply palm mask to edges
        edges = cv2.bitwise_and(edges, edges, mask=palm_mask)
        
        # 4. Define ROIs relative to Palm Center (cx, cy) and Radius (r)
        
        line_masks = {
            'heart': np.zeros_like(edges),
            'head': np.zeros_like(edges),
            'life': np.zeros_like(edges),
            'fate': np.zeros_like(edges)
        }
        
        # Helper to create ROI mask
        def get_roi_mask(pts):
            roi = np.zeros_like(mask)
            cv2.fillPoly(roi, [np.array(pts, np.int32)], 255)
            return roi

        # Heart Line: Upper part of palm
        # Above Head line, usually starts from ulnar side (pinky side)
        # Region: Above center, extending to the side opposite the thumb
        if side == 'right':
            # Thumb on Right -> Pinky on Left
            # Heart line is on the Left-Top quadrant of the palm
            pts_heart = [
                (cx - int(1.5*r), cy - int(1.2*r)), # Top-Left
                (cx + int(0.2*r), cy - int(1.2*r)), # Top-Right (near center)
                (cx + int(0.2*r), cy - int(0.2*r)), # Bottom-Right
                (cx - int(1.5*r), cy - int(0.2*r))  # Bottom-Left
            ]
        else:
            # Thumb on Left -> Pinky on Right
            # Heart line is on the Right-Top quadrant
            pts_heart = [
                (cx - int(0.2*r), cy - int(1.2*r)), # Top-Left
                (cx + int(1.5*r), cy - int(1.2*r)), # Top-Right
                (cx + int(1.5*r), cy - int(0.2*r)), # Bottom-Right
                (cx - int(0.2*r), cy - int(0.2*r))  # Bottom-Left
            ]
            
        # Head Line: Middle part of palm
        # Below Heart line, often starts with Life line on thumb side
        # Region: Central horizontal band
        if side == 'right':
            # Thumb on Right
            pts_head = [
                (cx - int(1.5*r), cy - int(0.4*r)), 
                (cx + int(0.8*r), cy - int(0.4*r)),
                (cx + int(0.8*r), cy + int(0.3*r)),
                (cx - int(1.5*r), cy + int(0.3*r))
            ]
        else:
            # Thumb on Left
            pts_head = [
                (cx - int(0.8*r), cy - int(0.4*r)),
                (cx + int(1.5*r), cy - int(0.4*r)),
                (cx + int(1.5*r), cy + int(0.3*r)),
                (cx - int(0.8*r), cy + int(0.3*r))
            ]

        # Life Line: Curves around thumb
        # Region: Thumb side, lower quadrant
        if side == 'right':
            # Thumb on Right
            pts_life = [
                (cx, cy - int(0.4*r)),
                (cx + int(1.5*r), cy - int(0.4*r)),
                (cx + int(1.5*r), cy + int(1.5*r)),
                (cx, cy + int(1.5*r))
            ]
        else:
            # Thumb on Left
            pts_life = [
                (cx - int(1.5*r), cy - int(0.4*r)),
                (cx, cy - int(0.4*r)),
                (cx, cy + int(1.5*r)),
                (cx - int(1.5*r), cy + int(1.5*r))
            ]
            
        # Fate Line: Vertical Center
        # Region: Central vertical strip
        pts_fate = [
            (cx - int(0.25*r), cy - int(0.5*r)),
            (cx + int(0.25*r), cy - int(0.5*r)),
            (cx + int(0.25*r), cy + int(1.5*r)),
            (cx - int(0.25*r), cy + int(1.5*r))
        ]

        # Create ROI masks
        heart_roi = get_roi_mask(pts_heart)
        head_roi = get_roi_mask(pts_head)
        life_roi = get_roi_mask(pts_life)
        fate_roi = get_roi_mask(pts_fate)
        
        # Further restrict ROIs to the palm mask
        heart_roi = cv2.bitwise_and(heart_roi, palm_mask)
        head_roi = cv2.bitwise_and(head_roi, palm_mask)
        life_roi = cv2.bitwise_and(life_roi, palm_mask)
        fate_roi = cv2.bitwise_and(fate_roi, palm_mask)

        # Helper to clean up lines
        def clean_lines(edge_img, roi):
            masked = cv2.bitwise_and(edge_img, edge_img, mask=roi)
            # Morphological closing to connect gaps
            kernel = np.ones((3,3), np.uint8)
            closed = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)
            # Remove small noise
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
            sizes = stats[1:, -1]; nb_components = nb_components - 1
            min_size = 20 # Minimum line length/area
            clean = np.zeros((output.shape), dtype=np.uint8)
            for i in range(0, nb_components):
                if sizes[i] >= min_size:
                    clean[output == i + 1] = 255
            return clean

        line_masks['heart'] = clean_lines(edges, heart_roi)
        line_masks['head'] = clean_lines(edges, head_roi)
        line_masks['life'] = clean_lines(edges, life_roi)
        line_masks['fate'] = clean_lines(edges, fate_roi)
        
        return line_masks

    def visualize_lines(
        self, 
        image: np.ndarray, 
        line_masks: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Overlay detected lines on the original image.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        colors = {
            'heart': (0, 0, 255),   # Red
            'head': (0, 255, 0),    # Green
            'life': (255, 0, 0),    # Blue
            'fate': (0, 255, 255)   # Yellow
        }
        
        for name, mask in line_masks.items():
            # Dilate mask to make lines visible
            kernel = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            
            # Overlay
            vis[dilated > 0] = colors.get(name, (255, 255, 255))
            
        return vis
