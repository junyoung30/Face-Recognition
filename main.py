from collections import defaultdict
import numpy as np

from configs import TripletConfig
from data import *
from models import *
from training import *
from utils import *


def main():

    config = TripletConfig(
        image_size = 224,
        embedding_size = 512,
        lr = 0.0001,
        margin = 0.5,
        batch_size = 128,

        init_epoch = 50,
        cil_epoch = 50,

        num_total_identities = 200,
        num_base_identities = 150,

        known_persons = 100,
        init_persons = 20,
        cil_persons = 20,

        inc_num_images = 5,  # sampling number

        root_path = '/path/to/dataset',
        resolution = 'High',

        S = ["S001", "S002", "S003", "S005"],
        L = ["L1", "L2", "L3", "L8"],
        E = ["E01", "E02"],
        C = ["C3", "C5", "C6", "C7", "C9"],

        test_transform=get_test_transform(224),
        tr_transform=get_train_transform(224)
    )

    identity_pool, _ = sample_identity_pool(
        root_path = config.root_path,
        resolution = config.resolution,
        n_identities = config.num_total_identities,
        seed = config.identity_seed
    )

    known_unknown_pool = identity_pool[:config.num_base_identities]

    all_img_paths, all_labels = build_image_dataset(
        root_path = config.root_path,
        resolution = config.resolution,
        identity_list = known_unknown_pool,
        S = config.S, 
        L = config.L, 
        E = config.E, 
        C = config.C
    )


    person_to_paths = defaultdict(list)
    for path, label in zip(all_img_paths, all_labels):
        person_to_paths[label].append(path)

    rng = np.random.default_rng(1004) # Data Shuffle
    new_person_to_paths = {}
    for pid, paths in person_to_paths.items():
        selected = rng.choice(paths, size=60, replace=False)
        new_person_to_paths[pid] = {
            'train': selected[:30].tolist(),
            'val': selected[30:].tolist()
        }

    # ============== Main Execute ==============
    save_folder = 'experiments/test'
    device = 'cuda:0'

    # SEED_LIST = [1, 34, 42, 99, 666, 717, 1234, 5656, 32, 81]
    SEED_LIST = [1, 34, 42, 99, 666]

    for SEED in SEED_LIST:
        data = prepare_cil_data(
            k_uk_persons = known_unknown_pool, 
            per_person_paths = new_person_to_paths, 
            config = config,
            split_seed = SEED
        )

        model_names = (
            [f'model0_SEED{SEED}.pth'] + 
            [f'model{i}_SEED{SEED}_S{config.inc_num_images}.pth' 
            for i in range(1,config.cil_step)]
        )

        trainer = TripletTrainer(
            config = config,
            device = device,
            save_folder = save_folder,
            data = data,
            model_names = model_names,
            sample_strategy = sample_kmeans_n_per_class,
            training_strategy = "replay",
            triplet_fn = create_hard_semihard_triplet(config.margin)
        )

        model = FaceNet_MobileNetV3Large(embedding_size=config.embedding_size)
        trainer.train_initial(model)

        for phase in range(1, config.cil_step):
            trainer.train_incremental(model, phase)



if __name__ == "__main__":
    main()