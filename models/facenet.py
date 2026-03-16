import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import math
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')



class FaceNet_MobileNetV2(nn.Module):
    def __init__(self, embedding_size:int, seed:int=42):
        super(FaceNet, self).__init__()
        base_model = models.mobilenet_v2(pretrained=True)
        
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        torch.manual_seed(seed)
        self.fc = nn.Linear(base_model.last_channel, embedding_size)
                
    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class FaceNet_MobileNetV3Small(nn.Module):
    def __init__(self, embedding_size:int, seed:int=42):
        super(FaceNet_MobileNetV3Small, self).__init__()
        base_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        torch.manual_seed(seed)
        self.fc = nn.Linear(base_model.classifier[0].in_features, embedding_size)
                
    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    
class FaceNet_MobileNetV3Large(nn.Module):
    def __init__(self, embedding_size:int, seed:int=42):
        super(FaceNet_MobileNetV3Large, self).__init__()
        base_model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        torch.manual_seed(seed)
        self.fc = nn.Linear(base_model.classifier[0].in_features, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class FaceNet_ShuffleNetV2(nn.Module):
    def __init__(self, embedding_size:int, variant:str="x1_0", seed:int=42):
        super(FaceNet_ShuffleNetV2, self).__init__()

        if variant == "x0_5":
            base_model = models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1")
        elif variant == "x1_0":
            base_model = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        elif variant == "x1_5":
            base_model = models.shufflenet_v2_x1_5(weights="IMAGENET1K_V1")
        elif variant == "x2_0":
            base_model = models.shufflenet_v2_x2_0(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported ShuffleNetV2 variant: {variant}")

        self.features = nn.Sequential(
            base_model.conv1,
            base_model.maxpool,
            base_model.stage2,
            base_model.stage3,
            base_model.stage4,
            base_model.conv5
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        torch.manual_seed(seed)
        # classifier[1].in_features = 마지막 FC input size
        self.fc = nn.Linear(base_model.fc.in_features, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    
class FaceNet_EfficientNet(nn.Module):
    def __init__(self, version: str = "b0", embedding_size: int = 512, seed: int = 42):

        super(FaceNet_EfficientNet, self).__init__()

        if version == "b0":
            base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_features = 1280
        elif version == "b1":
            base_model = models.efficientnet_b1(weights="IMAGENET1K_V1")
            in_features = 1280
        elif version == "b2":
            base_model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            in_features = 1408
        else:
            raise ValueError(f"Unknown EfficientNet version: {version}")

        # Feature extractor 부분만 사용 (classifier 제거)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        torch.manual_seed(seed)
        self.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x