from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class BaseConfig:
    root_path: str
    
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    image_size: int = 224
    resolution: str = "High"
            
    S: List[str] = field(default_factory=lambda: ["S001", "S002", "S003", "S005"])
    L: List[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L8"])
    E: List[str] = field(default_factory=lambda: ["E01", "E02"])
    C: List[str] = field(default_factory=lambda: ["C3", "C5", "C6", "C7", "C9"])
    
    num_total_identities: int = 200
    identity_seed: int = 1234
    num_base_identities: int = 150   # known + unknown
    
    known_persons: int = 100
    init_persons: int = 20
    cil_persons: int = 20
    
    batch_size: int = 128
    lr: float = 0.0001
    embedding_size: int = 512
    
    init_epoch: int = 100
    cil_epoch: int = 60
    
    num_batch_images: int = 4  # init batch sampler
    inc_num_images: int = 5  # number of sampling per class
        
    tr_transform: any = None
    test_transform: any = None
        
    fpir_targets: List[float] = (0.01, 0.001)
        
    def __post_init__(self):
        self.cil_step = (self.known_persons - self.init_persons) // self.cil_persons + 1
    
@dataclass
class TripletConfig(BaseConfig):
    backbone_name: str = "mobilenet_v2"
    margin: float = 0.5
    
@dataclass
class ArcFaceConfig(BaseConfig):
    backbone_name: str = "mobilenet_v2"
    scale: float = 32.0
    margin: float = 0.5
    num_classes: int = 1000