import torch
import torch.nn as nn
import timm

class RegionEncoder(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
        """
        Encoder for specific palm regions (Heart, Head, Life, Fate lines).
        Uses a smaller ViT by default to save resources.
        """
        super(RegionEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(0) # Remove classifier to get features

    def forward(self, x):
        return self.model(x)

class MultiRegionEncoder(nn.Module):
    def __init__(self, regions=['heart', 'head', 'life', 'fate']):
        super(MultiRegionEncoder, self).__init__()
        self.regions = regions
        self.encoders = nn.ModuleDict({
            region: RegionEncoder() for region in regions
        })

    def forward(self, region_crops):
        """
        Args:
            region_crops (dict): Dictionary of tensors for each region.
        Returns:
            dict: Dictionary of feature vectors for each region.
        """
        features = {}
        for region, crop in region_crops.items():
            if region in self.encoders:
                features[region] = self.encoders[region](crop)
        return features
