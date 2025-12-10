import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader



class TrainerBase:
    def __init__(self, config, device, save_folder):
        self.config = config
        self.device = device
        self.save_folder = save_folder
        
    def _make_train_loader(self, dataset, sampler=None):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(1234)
        
        if sampler:
            return DataLoader(
                dataset,
                batch_sampler = sampler,
                num_workers=self.config.num_workers, 
                pin_memory=self.config.pin_memory,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers, 
                pin_memory=self.config.pin_memory,
                worker_init_fn=seed_worker,
                generator=g
            )
    
    def _make_eval_loader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )

    def _save_model(
        self, model, metadata, best_tpir_model_states, model_name,
    ):
        os.makedirs(self.save_folder, exist_ok=True)
        save_path = os.path.join(self.save_folder, model_name)
        model.eval()
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_tpir_state_dict': best_tpir_model_states,
            'metadata': metadata
        }, save_path)