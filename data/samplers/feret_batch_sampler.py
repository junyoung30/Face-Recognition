import math
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict

class FERETBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_of_images=2, seed=1004):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_of_images = num_of_images
        self.seed = seed
        
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        
        epoch = getattr(self, "epoch", 0)
        rng = np.random.default_rng(self.seed + epoch)
        
        batch = []
        class_indices = {cls: indices[:] for cls, indices in self.class_to_indices.items()}
        current_classes = self.classes[:]
        
        while class_indices:
            selected_classes = rng.choice(
                current_classes,
                min(len(current_classes), int(self.batch_size / self.num_of_images)),
                replace=False
            )
            
            for cls in selected_classes:
                indices = class_indices[cls]
                if len(indices) >= self.num_of_images:
                    sampled_indices = rng.choice(
                        indices, 
                        self.num_of_images, 
                        replace=False
                    )
                    for idx in sampled_indices:
                        indices.remove(idx)
                    batch.extend(sampled_indices)
                    
                    if len(batch) >= self.batch_size:
                        yield batch[:self.batch_size]
                        batch = batch[self.batch_size:]

                if len(indices) < self.num_of_images:
                    del class_indices[cls]
                    current_classes.remove(cls)
        
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)