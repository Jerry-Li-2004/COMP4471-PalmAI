import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.palm_ai import PalmAI
from data.loader import PalmDataset
from data.preprocess import get_transforms

def train_model(data_dir, num_epochs=5, batch_size=4, learning_rate=1e-4):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data Preparation
    transform = get_transforms(input_size=224)
    dataset = PalmDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Found {len(dataset)} images in {data_dir}")
    
    # 3. Model Initialization
    model = PalmAI().to(device)
    
    # 4. Loss and Optimizer
    # Regression loss for score
    criterion_score = nn.MSELoss() 
    # For text, we'd use CrossEntropy if we had tokens, or MSE if embedding regression
    # Here we simulate embedding regression
    criterion_text = nn.MSELoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 5. Training Loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Dummy targets for demonstration
            # In real training, these would come from the dataset
            target_score = torch.rand(images.size(0), 1).to(device) # Random score 0-1
            target_text_emb = torch.randn(images.size(0), 768).to(device) # Random embedding
            
            optimizer.zero_grad()
            
            # Forward
            pred_score, pred_text_emb = model(images)
            
            # Loss
            loss_s = criterion_score(pred_score, target_score)
            loss_t = criterion_text(pred_text_emb, target_text_emb)
            loss = loss_s + loss_t
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 2 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {running_loss/len(dataloader):.4f}")
        
    # 6. Save Model
    save_path = os.path.join(os.path.dirname(__file__), '../output/palm_ai_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Use synthetic data by default
    data_path = os.path.join(os.path.dirname(__file__), '../data/synthetic')
    
    # Check if data exists
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        print("Data directory empty or missing. Please run generate_synthetic_data.py first.")
    else:
        train_model(data_path)
