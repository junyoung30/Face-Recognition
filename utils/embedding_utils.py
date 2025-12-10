import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from typing import Tuple

from utils.metrics import distance, DistanceMethod



def get_database(
    y_pred: torch.Tensor, 
    db_labels: torch.Tensor,
    dist_method: DistanceMethod = DistanceMethod.L2
) -> pd.DataFrame:
    """
    Params:
        y_pred: The embeddings for all images
        db_labels: The labels corresponding to y_pred
    
        examples)
                    emb1 label1
                    emb2 label2
                    ...
    Returns:
        DataFrame
    """

    person_to_embeddings = defaultdict(list)
    for emb, label in zip(y_pred, db_labels):
        person_to_embeddings[label].append(emb)
    
    data = {
        "person":[],
        "embedding_mean": [],
        "intra_class_distance": [],
    }
    

    for person, embeddings in person_to_embeddings.items():
        embeddings = torch.stack(embeddings)
        mean_emb = F.normalize(embeddings.mean(dim=0), dim=0).numpy()
        
        pairs = list(itertools.combinations(embeddings, 2))
        intra_dists = [
            distance(a.numpy(), b.numpy(), DistanceMethod.L2)
            for a, b in pairs
        ]
        intra_class = np.mean(intra_dists)

    
        data["person"].append(person)
        data["embedding_mean"].append(mean_emb)
        data["intra_class_distance"].append(intra_class)
    

    data["inter_class_distance"] = [None] * len(data["person"])
    db = pd.DataFrame(data)
    
    for i, row in db.iterrows():
        person_emb = row["embedding_mean"]
        others = [x for j, x in enumerate(db["embedding_mean"]) if j != i]
        others = np.vstack(others)
        dists = [distance(person_emb, o, dist_method) for o in others]
        db.at[i, "inter_class_distance"] = np.mean(dists)
    return db

@torch.no_grad() 
def get_embedding(
    model: nn.Module, 
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[Tensor, Tensor]:

    model.eval()
    
    y_pred_list = []
    label_list = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
            
        preds = model(images)
        y_pred_list.append(preds.cpu())
        label_list.append(labels.cpu())
    
    y_pred = torch.cat(y_pred_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return y_pred, labels