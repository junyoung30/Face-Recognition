import itertools
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional, Any

from utils import get_bbox



def sample_identity_pool(
    root_path: str,
    resolution: str, 
    n_identities: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    루트 경로에서 인물 디렉토리를 읽어 n_identities 만큼 무작위로 선택하여 반환함.
    
    Args:
        root_path: 인물 폴더가 저장된 경로 (Path 객체로 변환됨).
        resolution: 화질 ["High", "Middle", "Low"]
        n_identities: 선택할 인물 수.
        seed: 재현성을 위한 무작위 셔플 시드 값.
        
    Raises:
        ValueError: 요청한 인물 수가 전체 폴더 수보다 클 때 발생.
        
    Returns:
        selected : 선택된 인물 ID.
        non_selected : 선택되지 않은 인물 ID.
    """

    root_path = Path(root_path) / resolution
    rng = np.random.default_rng(seed)
    
    duplicate_ids = ["19082342", "19091733", "19062722"]
    
    all_person_dirs = [
        person_dir
        for person_dir in root_path.iterdir()
        if person_dir.is_dir() and person_dir.name not in duplicate_ids
    ]
    
    # Exception handling
    if n_identities > len(all_person_dirs):
        raise ValueError(f"Requested {n_identities}, but only {len(all_person_dirs)} available.")
    
    all_person_dirs = np.array(all_person_dirs)
    rng.shuffle(all_person_dirs)
    
    selected = np.array([p.name for p in all_person_dirs[:n_identities]])
    non_selected = np.array([p.name for p in all_person_dirs[n_identities:]])
    
    return selected, non_selected


def build_image_dataset(
    root_path: str,
    resolution: str,
    identity_list: np.ndarray,
    S: List[str],
    L: List[str],
    E: List[str],
    C: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    주어진 인물 ID와 속성 조건들을 조합하여 이미지 경로 및 레이블을 생성.
    
    Args:
        root_path: 이미지의 루트 디렉토리 경로.
        identity_list: 처리할 인물 ID가 담긴 배열.
        S: 악세서리 조건 목록.
        L: 조명 조건 목록.
        E: 표정 조건 목록.
        C: 각도 조건 목록.
    
    Returns:
        image_paths: 이미지 파일 경로들의 배열.
        labels: 각 이미지 경로에 대응되는 인물 ID 배열.
    """
    
    root_path = Path(root_path) / resolution
    
    image_paths = []
    labels = []
    
    for identity in identity_list:
        identity_dir = root_path / identity
        
        for s, l, e, c in itertools.product(S, L, E, C):
            image_path = identity_dir / s / l / e / f"{c}.jpg"
            image_paths.append(str(image_path))
            labels.append(identity)
            
    return np.array(image_paths), np.array(labels)


class KFaceDataset(Dataset):
    """
    KFace 데이터셋을 위한 커스텀 PyTorch Dataset 클래스.
    
    주어진 이미지 경로 목록을 기반으로,
    각 이미지에 대해 bounding box를 적용하여 crop한 뒤 반환함.
    클래스 인덱스는 인물 ID를 기준으로 자동 생성됨.
    
    Attributes:
        image_paths: 이미지 경로 객체 목록.
        resolution: 이미지 해상도 값.
        transform: 이미지에 적용할 전처리/Augmentation 함수.
        class_to_idx: 인물 ID와 정수 레이블 간의 매핑.
        classes: 정렬된 고유 인물 ID 목록.
        labels: 각 이미지 경로에 대응하는 정수 레이블 목록
    """
    def __init__(
        self, 
        image_paths: np.ndarray,
        root_path: str,
        resolution: str,
        transform: Optional[Any] = None
    ):
        """
        Args:
            image_paths: 이미지 파일 경로들이 담긴 배열.
            root_path: 데이터셋의 루트 디렉토리 경로.
            resolution: 사용할 해상도.
            transform: 이미지에 적용할 PyTorch Transform 함수.
        """
        
        self.image_paths = [Path(p) for p in image_paths]
        self.root_path = Path(root_path)
        self.resolution = resolution
        self.transform = transform
        
        # 고유 인물 ID 추출 및 정렬
        unique_ids = set()
        temp_labels = []
        
        for path in self.image_paths:
            person_id = path.parts[-5]
            unique_ids.add(person_id)
            temp_labels.append(person_id)
        
        self.classes = sorted(list(unique_ids))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.labels = [self.class_to_idx[pid] for pid in temp_labels]
        
        self.bbox_cache = {}
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def get_image(self, path: Path) -> Image.Image:
        """
        이미지를 로드하고 BBox에 맞춰 Crop하여 반환함.
        
        Args:
            path: 로드할 이미지의 경로.
        Returns:
            image: Crop된 PIL 이미지 객체 (RGB).
        """
        
        person_id = path.parts[-5]
        c = path.stem
        
        if person_id not in self.bbox_cache:
            self.bbox_cache[person_id] = get_bbox(
                self.resolution, person_id, self.root_path
            )
        bbox_dict = self.bbox_cache[person_id]
        bbox = bbox_dict[c]
        
        image = Image.open(path).convert('RGB')
        image = image.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
        return image
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image_path = self.image_paths[index]
        image = self.get_image(image_path)
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[index]
        return image, label