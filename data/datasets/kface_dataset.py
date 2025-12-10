import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset

from utils import get_bbox



def select_persons(
    root_path: str, 
    n_persons: int, 
    seed: int = 42
):

    root_path = Path(root_path)
    rng = np.random.default_rng(seed)
    
    duplicate_ids = ["19082342", "19091733", "19062722"]
    
    all_person_dirs = [
        person_dir
        for person_dir in root_path.iterdir()
        if person_dir.is_dir() and person_dir.name not in duplicate_ids
    ]
    
    # Exception handling
    if n_persons > len(all_person_dirs):
        raise ValueError(f"Requested {n_persons}, but only {len(all_person_dirs)} available.")
    
    all_person_dirs = np.array(all_person_dirs)
    rng.shuffle(all_person_dirs)
    
    selected = np.array([p.name for p in all_person_dirs[:n_persons]])
    non_selected = np.array([p.name for p in all_person_dirs[n_persons:]])
    
    return selected, non_selected


def get_image_paths_and_labels(
    root_path: str,
    selected_persons: np.ndarray,
    S: List['str'],
    L: List['str'],
    E: List['str'],
    C: List['str']
):
    
    root_path = Path(root_path)
    
    image_paths = []
    labels = []
    
    for person_id in selected_persons:
        person_dir = root_path / person_id
        
        for s, l, e, c in itertools.product(S, L, E, C):
            image_path = person_dir / s / l / e / (c + '.jpg')
            image_paths.append(str(image_path))
            labels.append(person_id)
            
    return np.array(image_paths), np.array(labels)


class KFaceDataset(Dataset):
    def __init__(self, image_paths, transform=None, resolution='High'):
        
        self.image_paths = image_paths
        self.transform = transform
        self.resolution = resolution
        self.base_path = Path("path/to/dataset")
        
        self.class_to_images = defaultdict(list)
        for image_path in image_paths:
            person = image_path.split('/')[-5]
            self.class_to_images[person].append(image_path)
            
        self.classes = list(self.class_to_images.keys())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.bbox_cache = {}
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_image(self, path):
        person_id = path.split('/')[-5]
        c = path.split('/')[-1].split('.')[-2]
        
        if person_id not in self.bbox_cache:
            self.bbox_cache[person_id] = get_bbox(
                self.resolution, person_id, self.base_path
            )
        bbox_dict = self.bbox_cache[person_id]
        bbox = bbox_dict[c]
        
        image = Image.open(path).convert('RGB')
        image = image.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
        return image
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.get_image(image_path)
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[image_path.split('/')[-5]]
        return image, label