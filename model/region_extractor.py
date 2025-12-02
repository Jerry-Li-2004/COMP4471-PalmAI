import torch
import torchvision.transforms.functional as F

class RegionExtractor:
    def __init__(self, method='fixed'):
        """
        Extracts regions of interest (ROI) from the palm image.
        Args:
            method (str): 'fixed' for fixed coordinates, 'yolo' for object detection.
        """
        self.method = method

    def extract(self, image, boxes=None):
        """
        Args:
            image (torch.Tensor): (C, H, W) image tensor.
            boxes (dict): Optional dictionary of bounding boxes {region: [x1, y1, x2, y2]}.
        Returns:
            dict: Dictionary of cropped tensors {region: crop_tensor}.
        """
        crops = {}
        _, h, w = image.shape
        
        if self.method == 'fixed' or boxes is None:
            # Define fixed relative coordinates (approximate)
            # These are placeholders and should be tuned
            default_boxes = {
                'heart': [0.1, 0.1, 0.9, 0.4], # Top part
                'head': [0.1, 0.3, 0.9, 0.6],  # Middle part
                'life': [0.1, 0.4, 0.6, 0.9],  # Curved around thumb
                'fate': [0.4, 0.4, 0.6, 0.9]   # Vertical center
            }
            
            for region, box in default_boxes.items():
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
                
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crops[region] = image[:, y1:y2, x1:x2]
                
                # Resize to expected encoder input (e.g., 224x224)
                crops[region] = F.resize(crops[region], [224, 224])
                
        return crops
