import torch
from torch.utils.data import DataLoader

from data.datasets.kface_dataset import KFaceDataset
from data.datasets.feret_dataset import FERETDataset

from utils import get_embedding



class ReplayBuffer:
    def __init__(
        self, 
        config, 
        strategy_func, 
        device,
        dataset_name,
    ):
        self.config = config
        self.device = device
        self.strategy_func = strategy_func
        self.dataset_name = dataset_name
        self.paths = []
        self.labels = []
        
    def sample(self, model, paths, labels):
        if self.dataset_name == "kface":
            ds = KFaceDataset(
                paths, self.config.test_transform, self.config.resolution
            )
        elif self.dataset_name == "feret":
            ds = FERETDataset(
                paths, labels, self.config.test_transform
            )
            
        loader = DataLoader(
            ds, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        embeddings, _ = get_embedding(model, loader, self.device)
        
        s_paths, s_labels = self.strategy_func(
            paths, labels, embeddings,
            n=self.config.inc_num_images
        )
        
        self.paths += list(s_paths)
        self.labels += list(s_labels)
    
    def get_all(self):
        return self.paths, self.labels
    
    def sample_new(self, model, paths, labels):
        if self.dataset_name == "kface":
            ds = KFaceDataset(
                paths, self.config.test_transform, self.config.resolution
            )
        elif self.dataset_name == "feret":
            ds = FERETDataset(
                paths, labels, self.config.test_transform
            )
            
        loader = DataLoader(
            ds, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        embeddings, _ = get_embedding(model, loader, self.device)
        
        s_paths, s_labels = self.strategy_func(
            paths, labels, embeddings,
            n=self.config.inc_num_images
        )
        
        return list(s_paths), list(s_labels)