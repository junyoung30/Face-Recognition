from dataclasses import dataclass, field
from typing import List
import math

@dataclass
class CILConfig:
    image_size: int = 224
    embedding_size: int = 512
    lr: float = 0.0001
    margin: float = 0.5
    scale: float = 30.0
        
    epoch: int = 50
    init_epoch: int = 60
    cil_epoch: int = 60
    stage1_epoch: int = 20
    stage2_epoch: int = 40
    
    init_persons: int = 10
    cil_persons: int = 10
    cil_step: int = field(init=False)    
    
    init_num_images: int = 4  # init batch sampler
    cil_num_images: int = 2  # cil batch sampler
    
    inc_num_images: int = 5
    inc_num_clusters: int = 5
    
    batch_size: int = 128
    
    fpir_targets: List[float] = (0.01, 0.001)
    
    use_persons: int = 200
    known_persons: int = 100
    unknown_persons: int = 100
    person_seed:int = 1234
        
    tr_transform: any = None
    test_transform: any = None
    
    resolution: str = "High"
    root_path: str = "/path/to/data"
        
    num_workers: int = 4
    pin_memory: bool = True
        
    T: float = 1.0
        
    S: List[str] = field(default_factory=lambda: ["S001", "S002", "S003", "S005"])
    L: List[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L8"])
    E: List[str] = field(default_factory=lambda: ["E01", "E02"])
    C: List[str] = field(default_factory=lambda: ["C3", "C5", "C6", "C7", "C9"])
        
    def __post_init__(self):
        self.cil_step = (self.known_persons-self.init_persons) // self.cil_persons + 1