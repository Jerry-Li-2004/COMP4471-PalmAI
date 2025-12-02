import cv2
import numpy as np
import os
import random

def generate_synthetic_palm(output_path, num_images=50):
    """
    Generates synthetic palm images with random lines simulating Heart, Head, Life, and Fate lines.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print(f"Generating {num_images} synthetic images in {output_path}...")
    
    for i in range(num_images):
        # 1. Create a skin-colored background
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:] = (180, 200, 230) # Approx skin color (BGR)
        
        # Add some noise/texture
        noise = np.random.randint(-20, 20, (512, 512, 3), dtype=np.int16)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # 2. Draw Lines (simulated)
        # Heart Line (Top, curved)
        start_pt = (random.randint(400, 500), random.randint(100, 150))
        end_pt = (random.randint(50, 150), random.randint(100, 150))
        cv2.line(img, start_pt, end_pt, (100, 120, 150), thickness=random.randint(2, 5))
        
        # Head Line (Middle, straight-ish)
        start_pt = (random.randint(400, 500), random.randint(200, 250))
        end_pt = (random.randint(50, 150), random.randint(250, 300))
        cv2.line(img, start_pt, end_pt, (100, 120, 150), thickness=random.randint(2, 5))
        
        # Life Line (Curved around thumb area - bottom left)
        # Simple curve simulation using polylines
        pts = np.array([[450, 250], [300, 350], [250, 500]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (100, 120, 150), thickness=random.randint(2, 5))
        
        # Fate Line (Vertical center)
        start_pt = (random.randint(250, 300), random.randint(400, 500))
        end_pt = (random.randint(250, 300), random.randint(150, 200))
        cv2.line(img, start_pt, end_pt, (100, 120, 150), thickness=random.randint(1, 4))
        
        # 3. Save
        filename = os.path.join(output_path, f"palm_{i:04d}.jpg")
        cv2.imwrite(filename, img)
        
    print("Generation complete.")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '../data/synthetic')
    generate_synthetic_palm(output_dir)
