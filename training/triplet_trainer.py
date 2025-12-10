import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .trainer_base import TrainerBase
from .replay_buffer import ReplayBuffer

from models.facenet import TripletLoss
from data.datasets.kface_dataset import KFaceDataset
from data.datasets.feret_dataset import FERETDataset
from data.samplers.kface_batch_sampler import KFaceBatchSampler
from data.samplers.feret_batch_sampler import FERETBatchSampler
from utils import get_embedding, get_database, find_best_tpir



def set_global_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
class TripletTrainer(TrainerBase):
    def __init__(
        self,
        config, 
        device,
        save_folder,
        data,
        model_names, 
        sample_strategy,
        training_strategy,
        dataset_name,
        triplet_fn
    ):
        super().__init__(config, device, save_folder)
        
        self.data = data
        self.model_names = model_names
        self.strategy = training_strategy # 'replay' or 'finetune' or 'full'
        self.dataset_name = dataset_name  # 'kface' or 'feret'
        self.triplet_fn = triplet_fn
        self.replay_buffer = ReplayBuffer(
            config, sample_strategy, device, self.dataset_name
        )
    
    def _val_paths_labels(self, phase):
        if phase == 0:
            known_paths = self.data['D_val_img_paths'][0]
            known_labels = self.data['D_val_labels'][0]
        else:
            known_paths = sum(self.data['D_val_img_paths'][:phase+1], [])
            known_labels = sum(self.data['D_val_labels'][:phase+1], [])
            
        val_paths = known_paths + self.data['uk_val_img_paths']
        val_labels = known_labels + self.data['uk_val_labels']
        return val_paths, val_labels
    
    def _train_phase(
        self, 
        model, 
        tr_paths, tr_labels,
        gallery_paths, gallery_labels,
        val_paths, val_labels,
        model_name,
        num_epoch
    ):
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = TripletLoss(margin=self.config.margin)
        
        if self.dataset_name == "kface":
            tr_ds = KFaceDataset(
                tr_paths, 
                self.config.tr_transform, 
                self.config.resolution
            )
            tr_sampler = KFaceBatchSampler(
                tr_ds, 
                batch_size=self.config.batch_size
            )
            gal_ds = KFaceDataset(
                gallery_paths, 
                self.config.test_transform, 
                self.config.resolution
            )
            val_ds = KFaceDataset(
                val_paths, 
                self.config.test_transform, 
                self.config.resolution
            )
            
        elif self.dataset_name == "feret":
            tr_ds = FERETDataset(
                tr_paths, 
                tr_labels, 
                self.config.tr_transform
            )
            tr_sampler = FERETBatchSampler(
                tr_ds, 
                batch_size=self.config.batch_size
            )
            gal_ds = FERETDataset(
                gallery_paths, 
                gallery_labels, 
                self.config.test_transform
            )
            val_ds = FERETDataset(
                val_paths, 
                val_labels, 
                self.config.test_transform
            )
            
        tr_loader = self._make_train_loader(tr_ds, tr_sampler)
        gallery_loader = self._make_eval_loader(gal_ds)
        val_loader = self._make_eval_loader(val_ds)

        best_tpirs = [0] * len(self.config.fpir_targets)
        best_tpir_model_states = [None] * len(self.config.fpir_targets)
        losses, tpir_lists = [], []
        
        for epoch in tqdm(range(num_epoch), desc=f"Training {model_name}"):
            tr_loader.batch_sampler.set_epoch(epoch)
            
            train_loss = train_epoch(
                model, 
                criterion, 
                optimizer, 
                tr_loader, 
                self.device, 
                self.triplet_fn
            )
            losses.append(train_loss)
                        
            gallery_embs, _ = get_embedding(model, gallery_loader, self.device)
            db = get_database(gallery_embs, gallery_labels)
            db_persons = list(dict.fromkeys(gallery_labels))
            
            val_embs, _ = get_embedding(model, val_loader, self.device)
            
            tpirs = []
            for i, ft in enumerate(self.config.fpir_targets):
                _, tpir = find_best_tpir(
                    val_embeddings=val_embs, 
                    db=db, 
                    val_labels=val_labels,
                    known_persons=db_persons,
                    fpir_target=ft
                )
                tpirs.append(tpir)
                
                if tpir > best_tpirs[i]:
                    best_tpirs[i] = tpir
                    best_tpir_model_states[i] = {
                        k:v.cpu().clone() 
                        for k, v in model.state_dict().items()
                    }
            
            tpir_lists.append(tpirs)
        
        metadata = {
            "loss": losses, 
            "tpir": tpir_lists,
        }
        self._save_model(model, metadata, best_tpir_model_states, model_name)
        
        
    def train_initial(self, model):
        print("\n=== Initial Phase ===")
        set_global_seed(1234)
        
        tr_paths  = self.data['D_tr_img_paths'][0]
        tr_labels = self.data['D_tr_labels'][0]
        
        gallery_paths  = tr_paths.copy()
        gallery_labels = tr_labels.copy()
        
        val_paths, val_labels = self._val_paths_labels(phase=0)
        
        model = model.to(self.device)
        
        self._train_phase(
            model,
            tr_paths, tr_labels,
            gallery_paths, gallery_labels,
            val_paths, val_labels,
            self.model_names[0],
            self.config.init_epoch
        )
    
    def train_incremental(self, model, phase, training_strategy=None):
        print(f"\n=== Incremental Phase{phase} ===")
        set_global_seed(1234)
        
        best_state_dict = torch.load(
            os.path.join(self.save_folder, self.model_names[phase-1])
        )['best_tpir_state_dict'][1]
        
        prev_best_model = model.to(self.device)
        prev_best_model.load_state_dict(best_state_dict)
        
        if training_strategy == "head_only":
            for p in prev_best_model.features.parameters():
                p.requires_grad = False
            for p in prev_best_model.fc.parameters():
                p.requires_grad = True
        
        if self.strategy == "replay":
            prev_paths  = self.data['D_tr_img_paths'][phase - 1]
            prev_labels = self.data['D_tr_labels'][phase - 1]

            self.replay_buffer.sample(prev_best_model, prev_paths, prev_labels)
            replay_paths, replay_labels = self.replay_buffer.get_all()
        
            current_paths  = self.data['D_tr_img_paths'][phase]
            current_labels = self.data['D_tr_labels'][phase]
        
            tr_paths  = replay_paths + current_paths
            tr_labels = replay_labels + current_labels
            gallery_paths  = tr_paths.copy()
            gallery_labels = tr_labels.copy()
        
        elif self.strategy == "finetune":
            tr_paths  = self.data['D_tr_img_paths'][phase]
            tr_labels = self.data['D_tr_labels'][phase]
            gallery_paths = sum(self.data['D_tr_img_paths'][:phase+1], [])
            gallery_labels = sum(self.data['D_tr_labels'][:phase+1], [])
            
        elif self.strategy == "full":
            tr_paths = sum(self.data['D_tr_img_paths'][:phase+1], [])
            tr_labels = sum(self.data['D_tr_labels'][:phase+1], [])
            gallery_paths = tr_paths.copy()
            gallery_labels = tr_labels.copy()
        
        val_paths, val_labels = self._val_paths_labels(phase=phase)
        
        self._train_phase(
            prev_best_model,
            tr_paths, tr_labels,
            gallery_paths, gallery_labels,
            val_paths, val_labels,
            self.model_names[phase],
            self.config.cil_epoch
        ) 