import torch
import sys
import os
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.palm_ai import PalmAI
from data.preprocess import get_transforms

def predict(image_path, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = PalmAI().to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print("No weights found, using random initialization for demo.")
    
    model.eval()
    
    # Preprocess
    transform = get_transforms(input_size=224)
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        score, text_emb = model(input_tensor)
        
    print(f"\n--- Inference Results for {os.path.basename(image_path)} ---")
    print(f"Predicted Score: {score.item():.4f}")
    print(f"Text Embedding Shape: {text_emb.shape}")
    # In a real app, we would decode text_emb here
    
    return score.item()

if __name__ == "__main__":
    # Example usage
    # Find a sample image
    data_dir = os.path.join(os.path.dirname(__file__), '../data/synthetic')
    if os.path.exists(data_dir):
        images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        if images:
            sample_img = os.path.join(data_dir, images[0])
            predict(sample_img, model_path=os.path.join(os.path.dirname(__file__), '../output/palm_ai_model.pth'))
        else:
            print("No images found to test.")
    else:
        print("Data directory not found.")
