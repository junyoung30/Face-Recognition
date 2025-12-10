import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

def split_known_unknown(
    total: np.ndarray, 
    n_known: int = 100, 
    seed:int = 1004
):
    rng = np.random.default_rng(seed)
    persons = total.copy()
    rng.shuffle(persons)
    return persons[:n_known], persons[n_known:]


def make_cil_partitions(
    known_persons: np.ndarray, 
    init_persons: int, 
    cil_persons: int, 
    cil_step: int
):
    initial = known_persons[:init_persons]
    incremental = known_persons[init_persons:]
    D = [initial]
    for i in range(cil_step-1):
        s = i * cil_persons
        e = s + cil_persons
        D.append(incremental[s:e])
    return D

def gather_paths_labels(
    person_list: np.ndarray, 
    per_person_paths: dict, 
    which: str = "train"
):
    paths, labels = [], []
    for pid in person_list:
        use_paths = per_person_paths[pid][which]
        paths.extend(use_paths)
        labels.extend([pid]*len(use_paths))
    return paths, labels

def prepare_cil_data(
    k_uk_persons: np.ndarray,
    per_person_paths: dict,  # new_person_to_paths
    config,
    split_seed: int,  # SEED (control)
):
    known_persons, unknown_persons = split_known_unknown(
        k_uk_persons, 
        config.known_persons, 
        split_seed
    )
    
    uk_val_img_paths, uk_val_labels = gather_paths_labels(
        unknown_persons, 
        per_person_paths, 
        'val'
    )
    
    D = make_cil_partitions(
        known_persons, 
        config.init_persons, 
        config.cil_persons, 
        config.cil_step
    )
    
    D_tr_img_paths, D_tr_labels = [], []
    D_val_img_paths, D_val_labels = [], []
    
    for persons in D:
        tr_p, tr_l = gather_paths_labels(persons, per_person_paths, 'train')
        va_p, va_l = gather_paths_labels(persons, per_person_paths, 'val')
        D_tr_img_paths.append(tr_p)
        D_tr_labels.append(tr_l)
        D_val_img_paths.append(va_p)
        D_val_labels.append(va_l)
        
    return {
        "uk_val_img_paths": uk_val_img_paths,
        "uk_val_labels": uk_val_labels,
        "D_tr_img_paths": D_tr_img_paths,
        "D_tr_labels": D_tr_labels,
        "D_val_img_paths": D_val_img_paths,
        "D_val_labels": D_val_labels,
        "known_persons": known_persons,
        "unknown_persons": unknown_persons
    }