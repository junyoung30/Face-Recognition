from enum import Enum

import numpy as np
import pandas as pd
import torch



class DistanceMethod(Enum):
    L2 = 'l2'
    COSINE = 'cosine'
    GEODESIC = 'geodesic'


def distance(a: np.ndarray, 
             b: np.ndarray, 
             method:DistanceMethod=DistanceMethod.L2) -> float:
    
    a = a.flatten()
    b = b.flatten()
    
    if method == DistanceMethod.L2:
        dist = np.sqrt(np.sum((a-b)**2))
        
    elif method == DistanceMethod.COSINE:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm==0 or b_norm==0:
            return 1.0
        dist = np.dot(a, b) / (a_norm * b_norm)
    
    elif method == DistanceMethod.GEODESIC:
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        dist = np.arccos(cos_sim)
    
    return dist


def evaluate_openset(
    db: pd.DataFrame,
    val_embeddings: torch.Tensor,
    val_labels: list,
    known_persons: list,
    threshold: float
):

    val_labels = np.array(val_labels)
    
    # ===== step 1 =====
    # DB의 평균 벡터 추출
    # ==================
    db_emb_mean = torch.stack([torch.tensor(vec.flatten())
                           for vec in db['embedding_mean']])  # (N_Class, D)
    
    # ===== step 2 =====
    # 임베딩 벡터들과 DB의 평균 벡터 사이의 거리 계산 및 최단 거리 인덱스 추출
    # ==================
    dists = torch.cdist(val_embeddings, db_emb_mean)  # (N, N_Class)
    min_dist, min_idx = torch.min(dists, dim=1)  # (N, ), (N, )
    
    # ===== step 3 =====
    # 예측된 클래스
    # ==================
    predicted_class = db.iloc[min_idx.numpy()]['person'].values  # (N, )
    
    # ===== step 4 =====
    # known / unknown 분리
    # ==================
    known_set = set(known_persons)
    unknown_mask = torch.tensor([label not in known_set for label in val_labels])
    known_mask = ~unknown_mask

    known_dists = dists[known_mask]  # (N_known, N_Class)
    unknown_dists = dists[unknown_mask]  # (N_unknown, N_Class)
    
    # ===== step 5 =====
    # TPIR 계산 : threshold보다 가깝고, predicted label == ground-truth
    # ==================
    predicted_class_known = predicted_class[known_mask.numpy()]  # (N_known, )
    known_labels = val_labels[known_mask.numpy()]                # (N_known, )

    correct_match = (known_dists.min(dim=1).values < threshold) & (predicted_class_known == known_labels)
    tpir = correct_match.float().sum() / known_mask.sum()
    
    # ===== step 6 =====
    # FPIR 계산 : threshold보다 가까우면 오탐
    # ==================
    false_match = (unknown_dists.min(dim=1).values < threshold)  # (N_unknown, )
    fpir = false_match.float().sum() / unknown_mask.sum()

    return tpir.item(), fpir.item()


def find_best_tpir(
    db: pd.DataFrame,
    val_embeddings: torch.Tensor,
    val_labels: list, 
    known_persons: list, 
    fpir_target: float = 0.01,
    threshold_range: tuple = (0.0, 2.0),
    num_steps: int = 500
):

    thresholds = np.linspace(
        threshold_range[0], threshold_range[1], num_steps
    )
    
    best_tpir = 0.0
    best_threshold = None
    
    for th in thresholds:
        tpir, fpir = evaluate_openset(
            db=db, 
            val_embeddings=val_embeddings,
            val_labels=val_labels, 
            known_persons=known_persons, 
            threshold=th
        )
        if fpir <= fpir_target and tpir > best_tpir:
            best_tpir = tpir
            best_threshold = th
    
    return best_threshold, best_tpir