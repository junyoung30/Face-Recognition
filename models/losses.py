import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import math
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class TripletLoss(nn.Module):
    def __init__(self, margin:float=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, 
                anchor: Tensor, 
                positive: Tensor, 
                negative: Tensor) -> float:
        
        pos_dist = torch.sum((anchor - positive)**2, axis=1)
        neg_dist = torch.sum((anchor - negative)**2, axis=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()