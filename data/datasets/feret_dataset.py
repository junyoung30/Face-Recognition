import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

def correct_feret_img_paths(root_dir, resolution):
    
    image_paths = []
    labels = []

    for dvd in ['dvd1', 'dvd2']:
        if resolution == 'high':
            dvd_path = os.path.join(root_dir, dvd, 'data', 'images')
        elif resolution == 'middle':
            dvd_path = os.path.join(root_dir, dvd, 'data', 'smaller')
        elif resolution == 'low':
            dvd_path = os.path.join(root_dir, dvd, 'data', 'thumbnails')

        person_dirs = sorted(os.listdir(dvd_path))
        for person_id in person_dirs:
            person_path = os.path.join(dvd_path, person_id)
            if not os.path.isdir(person_path):
                continue

            ppm_files = glob(os.path.join(person_path, '*.ppm'))
            image_paths.extend(ppm_files)
            labels.extend([person_id] * len(ppm_files))
    
    return image_paths, labels

class FERETDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
        self.classes = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.labels = [self.label_to_idx[l] for l in labels]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label