import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, global_dim=768, local_dim=192, regions=['heart', 'head', 'life', 'fate']):
        """
        Fuses global backbone features with local region features.
        Args:
            global_dim (int): Dimension of the global backbone features (e.g., 768 for ViT-Base).
            local_dim (int): Dimension of the local encoder features (e.g., 192 for ViT-Tiny).
        """
        super(FeatureFusion, self).__init__()
        self.regions = regions
        self.total_dim = global_dim + (len(regions) * local_dim)
        
        # Simple concatenation followed by an MLP to fuse features
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )

    def forward(self, global_features, local_features):
        """
        Args:
            global_features (torch.Tensor): (B, global_dim)
            local_features (dict): Dictionary of (B, local_dim) tensors.
        """
        # Concatenate all local features
        local_cat = []
        for region in self.regions:
            if region in local_features:
                local_cat.append(local_features[region])
            else:
                # Handle missing regions if necessary (e.g., with zeros)
                pass 
        
        local_cat = torch.cat(local_cat, dim=1)
        
        # Concatenate global and local
        combined = torch.cat([global_features, local_cat], dim=1)
        
        fused = self.fusion_layer(combined)
        return fused
