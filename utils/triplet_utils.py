import numpy as np
from typing import Tuple, List
import itertools
import torch
import random
from torch import Tensor
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')



def create_hard_semihard_triplet(margin: float,
                                 positive_strategy: str = "random"):
    """
    Args:
        margin (float): Triplet margin
        positive_strategy (str): Strategy to select positives 
                                 ['random', 'hard', 'easy']
    """
    def get_hard_semihard_triplet(embeddings: Tensor, 
                                  labels: Tensor, 
                                  ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        triplets = []
        
        for i in range(len(labels)):
            anchor = embeddings[i]
            anchor_label = labels[i]
            pos_indices = [j for j in range(len(labels))
                           if labels[j] == anchor_label and j != i]
            neg_indices = [j for j in range(len(labels))
                           if labels[j] != anchor_label]
            
            if not pos_indices or not neg_indices:
                continue
            
            ## Select Positive (Random, Hard, Easy)
            pos_dists = distance_matrix[i, pos_indices]
            if positive_strategy == "random":
                pos_idx = random.choice(pos_indices)
            elif positive_strategy == "hard":
                pos_idx = pos_indices[torch.argmax(pos_dists).item()]
            elif positive_strategy == "easy":
                pos_idx = pos_indices[torch.argmin(pos_dists).item()]
            else:
                raise ValueError("Invalid positive strategy.")
            
            positive = embeddings[pos_idx]
            pos_dist = distance_matrix[i, pos_idx]
            
            ## Select Negative
            neg_dists = distance_matrix[i, neg_indices]
            
            # Hard Negative
            hard_negatives = [
                j 
                for j in range(len(neg_dists)) 
                if neg_dists[j] < pos_dist]
            # Semi-hard Negative
            semi_hard_negatives = [
                j 
                for j in range(len(neg_dists))
                if pos_dist < neg_dists[j] < pos_dist + margin
            ]
            
            if hard_negatives:
                hardest_neg_idx_relative = torch.argmin(neg_dists[hard_negatives]).item()
                hard_neg_idx = neg_indices[hard_negatives[hardest_neg_idx_relative]]
                hard_neg = embeddings[hard_neg_idx]
                triplets.append((anchor, positive, hard_neg))
            elif semi_hard_negatives:
                semi_hard_neg_idx_relative = torch.argmin(neg_dists[semi_hard_negatives]).item()
                semi_hard_neg_idx = neg_indices[semi_hard_negatives[semi_hard_neg_idx_relative]]
                semi_hard_neg = embeddings[semi_hard_neg_idx]
                triplets.append((anchor, positive, semi_hard_neg))
            else:
                easy_neg_idx = neg_indices[torch.argmin(neg_dists)]
                easy_neg = embeddings[easy_neg_idx]
                triplets.append((anchor, positive, easy_neg))
            
        return triplets
    return get_hard_semihard_triplet



def create_semihard_triplet(margin: float,
                            positive_strategy: str = "random"):
    """
    Args:
        margin (float): Triplet margin
        positive_strategy (str): Strategy to select positives 
                                 ['random', 'hard', 'easy']
    """
    def get_semihard_triplet(embeddings: Tensor, 
                             labels: Tensor, 
                             ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        triplets = []
        
        for i in range(len(labels)):
            anchor = embeddings[i]
            anchor_label = labels[i]
            pos_indices = [j for j in range(len(labels))
                           if labels[j] == anchor_label and j != i]
            neg_indices = [j for j in range(len(labels))
                           if labels[j] != anchor_label]
            
            if not pos_indices or not neg_indices:
                continue
            
            # Select Positive (Random, Hard, Easy)
            pos_dists = distance_matrix[i, pos_indices]
            if positive_strategy == "random":
                pos_idx = random.choice(pos_indices)
            elif positive_strategy == "hard":
                pos_idx = pos_indices[torch.argmax(pos_dists).item()]
            elif positive_strategy == "easy":
                pos_idx = pos_indices[torch.argmin(pos_dists).item()]
            else:
                raise ValueError("Invalid positive strategy.")
            
            positive = embeddings[pos_idx]
            pos_dist = distance_matrix[i, pos_idx]
            
            # Only use semi-hard negatives
            semi_hard_negatives = [j for j in neg_indices if pos_dist < distance_matrix[i, j] < pos_dist + margin]
            
            if semi_hard_negatives:
                semi_dists = [distance_matrix[i, j] for j in semi_hard_negatives]
                semi_idx = semi_hard_negatives[torch.argmin(torch.tensor(semi_dists)).item()]
                triplets.append((anchor, positive, embeddings[semi_idx]))
            # If no semi-hard negative, skip this triplet
            else:
                continue
            
        return triplets
    return get_semihard_triplet