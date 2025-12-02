import torch
import torch.nn as nn
import timm

class VisionTransformerBackbone(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(VisionTransformerBackbone, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Remove the classification head to get features
        self.model.reset_classifier(0)
        
    def forward(self, x):
        features = self.model(x)
        return features
