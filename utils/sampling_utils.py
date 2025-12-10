import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import defaultdict

def sample_kmeans_n_per_class(img_paths, labels, embeddings, n=5):
    class_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_to_indices[label].append(i)

    selected_paths = []
    selected_labels = []

    for label, idxs in class_to_indices.items():
        class_embs = embeddings[idxs]
        class_paths = [img_paths[i] for i in idxs]

        n_clusters = min(n, len(class_embs))
        data_np = (
            class_embs.detach().cpu().numpy().astype(np.float64)
            if isinstance(class_embs, torch.Tensor) 
            else np.asarray(class_embs, dtype=np.float64)
        )
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data_np)
        
        centers = kmeans.cluster_centers_
        labels_k = kmeans.labels_

        for k in range(n_clusters):
            cluster_mask = (labels_k == k)
            cluster_embs = class_embs[cluster_mask]
            cluster_paths = np.array(class_paths)[cluster_mask]
            if len(cluster_embs) == 0:
                continue

            center_tensor = torch.tensor(centers[k])
            dists = torch.norm(cluster_embs - center_tensor, dim=1)
            closest_idx = torch.argmin(dists).item()

            selected_paths.append(cluster_paths[closest_idx])
            selected_labels.append(label)

    return selected_paths, selected_labels

def sample_random_n_per_class(img_paths, labels, embeddings, n=5):
    base_seed = 1
    
    class_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_to_indices[label].append(i)

    selected_paths, selected_labels = [], []

    for i, (label, idxs) in enumerate(class_to_indices.items()):
        class_seed = base_seed + i
        rng = np.random.default_rng(class_seed)

        chosen = rng.choice(idxs, size=min(n, len(idxs)), replace=False)
        for i in chosen:
            selected_paths.append(img_paths[i])
            selected_labels.append(label)

    return selected_paths, selected_labels

def sample_icarl_herding_n_per_class(img_paths, labels, embeddings, n=5):
    class_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_to_indices[label].append(i)

    selected_paths, selected_labels = [], []

    for label, idxs in class_to_indices.items():
        class_embs = embeddings[idxs]
        mu = class_embs.mean(dim=0)
        selected_idx = []
        selected_sum = torch.zeros_like(mu)

        num_select = min(n, len(class_embs))

        for k in range(1, num_select + 1):
            best_idx = None
            best_dist = float("inf")

            for i in range(len(class_embs)):
                if i in selected_idx:
                    continue
                candidate_mean = (selected_sum + class_embs[i]) / k
                dist = torch.norm(mu - candidate_mean)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            selected_idx.append(best_idx)
            selected_sum += class_embs[best_idx]

        for i in selected_idx:
            selected_paths.append(img_paths[idxs[i]])
            selected_labels.append(label)

    return selected_paths, selected_labels