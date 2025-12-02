import torch
import torch.nn as nn
from .backbone import VisionTransformerBackbone
from .encoders import MultiRegionEncoder
from .fusion import FeatureFusion
from .heads import ScoringHead, TextDecoder
from .region_extractor import RegionExtractor

class PalmAI(nn.Module):
    def __init__(self):
        super(PalmAI, self).__init__()
        
        # 1. Backbone
        self.backbone = VisionTransformerBackbone(model_name='vit_base_patch16_224')
        
        # 2. Region Extractor (Logic, not NN module)
        self.region_extractor = RegionExtractor(method='fixed')
        
        # 3. Region Encoders
        self.region_encoders = MultiRegionEncoder(regions=['heart', 'head', 'life', 'fate'])
        
        # 4. Fusion
        # ViT-Base features: 768, ViT-Tiny features: 192
        self.fusion = FeatureFusion(global_dim=768, local_dim=192)
        
        # 5. Heads
        self.scoring_head = ScoringHead(input_dim=512)
        self.text_decoder = TextDecoder(input_dim=512)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image batch (B, C, H, W)
        """
        # Global Features
        global_features = self.backbone(x) # (B, 768)
        
        # Extract Regions (This part might need to be done outside if using non-differentiable crop)
        # For end-to-end, we assume crops are tensors. 
        # Here we iterate over batch. Ideally, this should be batched.
        # For simplicity in this prototype, we'll assume the extractor handles single images
        # or we pre-process crops. 
        # Let's assume 'x' is the full image and we crop on the fly.
        
        batch_crops = {r: [] for r in ['heart', 'head', 'life', 'fate']}
        for i in range(x.size(0)):
            crops = self.region_extractor.extract(x[i])
            for r in batch_crops:
                batch_crops[r].append(crops[r])
        
        # Stack crops: (B, C, 224, 224)
        for r in batch_crops:
            batch_crops[r] = torch.stack(batch_crops[r]).to(x.device)
            
        # Local Features
        local_features = self.region_encoders(batch_crops) # Dict of (B, 192)
        
        # Fusion
        fused_features = self.fusion(global_features, local_features) # (B, 512)
        
        # Outputs
        score = self.scoring_head(fused_features)
        text_embedding = self.text_decoder(fused_features)
        
        return score, text_embedding
