import cv2
import os

image_path = 'data/MALE/IMG_0001.JPG'
if os.path.exists(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        print(f"Image shape: {img.shape}")
    else:
        print("Failed to load image")
else:
    print("Image not found")
