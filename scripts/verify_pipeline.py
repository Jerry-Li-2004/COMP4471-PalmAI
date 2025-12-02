import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.palm_ai import PalmAI

def test_pipeline():
    print("Initializing PalmAI model...")
    model = PalmAI()
    model.eval()
    
    # Create dummy input: Batch size 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    print("Running forward pass...")
    try:
        score, text_embedding = model(dummy_input)
        print("Forward pass successful!")
        print(f"Score output shape: {score.shape}")
        print(f"Text embedding output shape: {text_embedding.shape}")
        
        # Check shapes
        assert score.shape == (2, 1), f"Expected score shape (2, 1), got {score.shape}"
        # Text embedding shape depends on the decoder projection, currently 768
        assert text_embedding.shape == (2, 768), f"Expected text embedding shape (2, 768), got {text_embedding.shape}"
        
        print("\nVerification PASSED.")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
