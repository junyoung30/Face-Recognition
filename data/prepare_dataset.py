import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class CILDataContainer:
    """CIL 데이터셋 분할 결과를 저장하는 컨테이너"""
    
    uk_val_img_paths: List[str]
    uk_val_labels: List[Any]
    
    D_tr_img_paths: List[List[str]] = field(default_factory=list)
    D_tr_labels: List[List[Any]] = field(default_factory=list)
    D_val_img_paths: List[List[str]] = field(default_factory=list)
    D_val_labels: List[List[Any]] = field(default_factory=list)
    
    known_persons: np.ndarray = field(default_factory=lambda: np.array([]))
    unknown_persons: np.ndarray = field(default_factory=lambda: np.array([]))

def split_known_unknown(
    total_persons: np.ndarray, 
    n_known: int = 100, 
    seed:int = 1004
) -> Tuple[np.ndarray, np.ndarray]:
    """
    전체 인물을 Known(학습 대상)과 Unknown(Open-set 테스트용)으로 분리.
    
    Args:
        total_persons: 전체 인물 ID 배열.
        n_known: Known 인물 수.
        seed: 셔플 시드.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Known 인물 배열, Unknown 인물 배열)
    """
    rng = np.random.default_rng(seed)
    
    persons = total_persons.copy()
    rng.shuffle(persons)
    
    if n_known > len(persons):
        raise ValueError(f"n_known({n_known}) cannot be larger than total persons({len(total_persons)}).")
        
    return persons[:n_known], persons[n_known:]


def make_cil_partitions(
    known_persons: np.ndarray, 
    n_init: int, 
    n_inc: int, 
    total_steps: int
) -> List[np.ndarray]:
    """
    Known 인물들을 주어진 phase 수에 따라 분할.
    
    Args:
        known_persons: Known의 인물 ID 배열.
        n_init: Phase 0의 인물 수.
        n_inc: Phase k의 인물 수.
        total_steps: 전체 phase 수 (초기 phase 포함).
    """
    required = n_init + n_inc * (total_steps - 1)
    if required > len(known_persons):
        raise ValueError("Not enough known_persons for given CIL configuration.")
    
    initial_phase = known_persons[:n_init]
    incremental_phase = known_persons[n_init:]
    
    D = [initial_phase]
    
    for i in range(total_steps - 1):
        start = i * n_inc
        end = start + n_inc
        D.append(incremental_phase[start:end])
    return D

def gather_paths_labels(
    person_list: np.ndarray, 
    per_person_paths: dict, 
    split_mode: str = "train"
):
    """
    특정 인물 리스트에 해당하는 이미지 경로와 레이블 수집.
    
    Args:
        person_list: 데이터를 수집할 인물 ID 배열.
        per_person_paths: {인물ID: {'train': [경로...], 'val': [경로...]}} 구조.
        split_mode: 'train' 또는 'val'
    
    Returns:
        Tuple[List[str], List[Any]]: (이미지 경로 리스트, 레이블 리스트)
    """
    
    paths, labels = [], []
    for pid in person_list:
        if pid not in per_person_paths:
            continue
            
        use_paths = per_person_paths[pid][split_mode]
        paths.extend(use_paths)
        labels.extend([pid]*len(use_paths))
    return paths, labels

def prepare_cil_data(
    k_uk_persons: np.ndarray,
    per_person_paths: dict,  # new_person_to_paths
    config: Any,
    split_seed: int,  # SEED (control)
):
    """
    CIL 실험을 위한 전체 데이터셋 파티션을 생성.
    
    Args:
        k_uk_persons: 전체 인물 ID 목록.
        per_person_paths: 인물별 이미지 경로 딕셔너리.
        config: 실험 설정 객체
        split_seed: Known/Unknown 분할 시드.
        
    Returns:
        
    """
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
    
    container = CILDataContainer(
        uk_val_img_paths=uk_val_img_paths,
        uk_val_labels=uk_val_labels,
        known_persons=known_persons,
        unknown_persons=unknown_persons
    )
    
    for persons_in_phase in D:
        tr_p, tr_l = gather_paths_labels(persons_in_phase, per_person_paths, 'train')
        container.D_tr_img_paths.append(tr_p)
        container.D_tr_labels.append(tr_l)
        
        va_p, va_l = gather_paths_labels(persons_in_phase, per_person_paths, 'val')
        container.D_val_img_paths.append(va_p)
        container.D_val_labels.append(va_l)
        
    return container