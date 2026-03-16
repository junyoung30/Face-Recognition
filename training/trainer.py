import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple
from torch import Tensor  
import torch.nn.functional as F



def train_epoch(model: nn.Module,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                train_loader:DataLoader,
                device:torch.device,
                triplet_selector: Callable[[Tensor, Tensor], List[Tuple[Tensor, Tensor, Tensor]]]
               ) -> float:
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        embeddings = model(images)
        triplets = triplet_selector(embeddings, labels)
        if not triplets:
            continue
            
        anchor_batch, pos_batch, neg_batch = zip(*triplets)
        anchor_batch = torch.stack(anchor_batch)
        pos_batch = torch.stack(pos_batch)
        neg_batch = torch.stack(neg_batch)
        
        loss = criterion(anchor_batch, pos_batch, neg_batch)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss / len(train_loader)
    return train_loss