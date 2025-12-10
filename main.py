import os
import numpy as np
import torch
from collections import defaultdict

from config import config
from data.datasets.kface_dataset import select_persons, get_image_paths_and_labels
from data.prepare_dataset import prepare_cil_data
from training.triplet_trainer import TripletTrainer
from models.facenet import FaceNet_MobileNetV3Large
from utils import (
    get_train_transform, 
    get_test_transform, 
    sample_kmeans_n_per_class, 
    create_hard_semihard_triplet
)


def main():

    config = CILConfig(
        image_size = 224,
        embedding_size = 512,
        lr = 0.0001,
        margin = 0.4,
        batch_size = 128,

        init_epoch = 100,
        cil_epoch = 60,

        init_persons = 50,
        cil_persons = 50,

        inc_num_images = 5,
        
        root_path = '/path/to/dataset',
        S = ["S001", "S002", "S003", "S005"],
        L = ["L1", "L2", "L3", "L8"],
        E = ["E01", "E02"],
        C = ["C3", "C5", "C6", "C7", "C9"],
        
        test_transform=get_test_transform(224),
        tr_transform=get_train_transform(224)
    )

    total_persons, _ = select_persons(
        root_path = config.root_path, 
        n_persons = config.use_persons,
        seed = config.person_seed
    )

    k_uk_persons = total_persons[:150]

    all_img_paths, all_labels = get_image_paths_and_labels(
        root_path = config.root_path, 
        selected_persons = k_uk_persons,
        S = config.S, 
        L = config.L, 
        E = config.E, 
        C = config.C
    )

    person_to_paths = defaultdict(list)
    for path, label in zip(all_img_paths, all_labels):
        person_to_paths[label].append(path)

    rng = np.random.default_rng(1004)
    new_person_to_paths = {}
    for pid, paths in person_to_paths.items():
        selected = rng.choice(paths, size=60, replace=False)
        new_person_to_paths[pid] = {
            'train': selected[:30].tolist(),
            'val': selected[30:].tolist()
        }


    # ========== Train ==========
    SEED_LIST = [1, 34, 42, 99, 666]

    for SEED in SEED_LIST:

        data = prepare_cil_data(
            k_uk_persons=k_uk_persons,
            per_person_paths=new_person_to_paths,
            config=config,
            split_seed=SEED
        )

        model_names = (
            [f'model0_SEED{SEED}.pth'] +
            [f'model{i}_SEED{SEED}_S{config.inc_num_images}.pth'
             for i in range(1, config.cil_step)]
        )

        trainer = TripletTrainer(
            config=config,
            device='cuda:0',
            save_folder='experiments/',
            data=data,
            model_names=model_names,
            sample_strategy=sample_kmeans_n_per_class,
            training_strategy='replay',
            dataset_name='kface',
            triplet_fn=create_hard_semihard_triplet(config.margin)
        )

        model = FaceNet_MobileNetV3Large(config.embedding_size)
        trainer.train_initial(model)
        
        for phase in range(1, config.cil_step):
            trainer.train_incremental(model, phase)



if __name__ == "__main__":
    main()