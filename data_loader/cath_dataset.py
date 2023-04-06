import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from config.config import Config
from data_loader.geometric_features import ProteinFeatures

N_CLASSES = 1453
N_LEVEL_CLASSES = [5, 26, 518]


def dataset_path():
    path_ = Path(__file__).parent.parent / 'dataset' / 'cath-domain-list-S35.txt'
    return path_


def train_test_path():
    path_ = Path(__file__).parent.parent / 'dataset'
    metadata_file = path_ / 'metadata.pkl'
    train_file = path_ / 'train_ids.pkl'
    test_file = path_ / 'test_ids.pkl'
    return metadata_file, train_file, test_file


class CATHData(object):
    def __init__(self):
        self.config = Config()
        data_path = dataset_path()
        assert os.path.exists(data_path)
        self.preprocessed_dir = Path(self.config.preprocessed_folder)
        self.cath_data = {}
        self.cath_cls = []
        self.parse_dataset(data_path)

    def parse_dataset(self, data_path):
        raw_df = pd.read_csv(str(data_path),
                             header=None,
                             comment='#',
                             delim_whitespace=True)
        df_cath_family = raw_df[1].map(str) + '.' + raw_df[2].map(str) + '.' \
                         + raw_df[3].map(str) + '.' + raw_df[4].map(str)
        df_cath_family = pd.concat([raw_df[0], df_cath_family], axis=1)
        df_cath_family.reset_index()
        df_cath_family = df_cath_family.set_axis([0, 1], axis=1, copy=True)
        file_ids = df_cath_family[0].tolist()
        cath_ids = df_cath_family[1].tolist()
        assert len(file_ids) == len(cath_ids)

        exist_cath_ids = []

        for i in range(len(file_ids)):
            file_rec = file_ids[i]
            pdb_id = file_rec[:4]
            chain = file_rec[4]
            file_key = f'{pdb_id}_{chain}'
            pdb_file_path = self.preprocessed_dir / f'{file_key}.pkl'

            if not os.path.exists(pdb_file_path):
                continue

            cath_id = cath_ids[i]

            if file_key in self.cath_data:
                self.cath_data[file_key].append(cath_id)
            else:
                self.cath_data.update({file_key: [cath_id]})

            exist_cath_ids.append(cath_id)

        self.cath_cls = list(set(exist_cath_ids))

    def train_test_split(self):
        assert self.cath_cls is not None
        assert self.cath_data is not None

        dataset = list(self.cath_data.keys())
        np.random.shuffle(dataset)
        train_size = int(len(dataset) * self.config.train_split_factor)
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]

        print(f'Dataset split: Train set {len(train_set)} Test set: {len(test_set)}')
        return train_set, test_set

    def save_dataset(self):
        metadata_file, train_file, test_file = train_test_path()
        train_set, test_set = self.train_test_split()

        with open(metadata_file, 'wb') as fh:
            metadata_rec = (self.cath_data, self.cath_cls)
            pickle.dump(metadata_rec, fh)

        with open(train_file, 'wb') as fh:
            pickle.dump(train_set, fh)

        with open(test_file, 'wb') as fh:
            pickle.dump(test_set, fh)

    def get_item_data(self, key):
        cath_cls = self.cath_data[key]
        cath_ids = [self.cath_cls.index(cl) for cl in cath_cls]
        return cath_ids


class CATHDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        config = Config()
        metadata_path, train_set_path, test_set_path = train_test_path()
        assert os.path.exists(metadata_path)
        assert os.path.exists(train_set_path)
        assert os.path.exists(test_set_path)
        self.preprocessed_dir = Path(config.preprocessed_folder)

        with open(metadata_path, 'rb') as fh:
            self.cath_data, cath_classes = pickle.load(fh)

        self.global_classes = []

        for cl in cath_classes:
            spl_ = cl.split('.')
            self.global_classes.append(f'{spl_[0]}.{spl_[1]}.{spl_[2]}')

        self.global_classes = list(set(self.global_classes))
        self.global_classes.sort()

        items_path = train_set_path if mode == 'train' else test_set_path

        with open(items_path, 'rb') as fh:
            self.items = pickle.load(fh)

        self.cath_levels = {i: set() for i in range(3)}

        for cl in cath_classes:
            cl_levels = cl.split('.')[:-1]
            for i, level in enumerate(cl_levels):
                self.cath_levels[i].add(int(level))

        for i, j in self.cath_levels.items():
            self.cath_levels.update({i: sorted(j)})

        self.protein_features = ProteinFeatures()

    def __getitem__(self, index):
        file_key = self.items[index]
        filepath = self.preprocessed_dir / f'{file_key}.pkl'

        with open(filepath, 'rb') as fh:
            coo, sequence, _ = pickle.load(fh)

        features = self.protein_features.build_features((coo, sequence))
        cath_cls = self.cath_data[file_key]
        target_cath = []
        for cl in cath_cls:
            c = cl.split('.')
            cath3 = f'{c[0]}.{c[1]}.{c[2]}'
            target_cath.append(cath3)
        target = self.get_cath_target(target_cath)
        return features, target

    def get_cath_target(self, cath_cls):
        global_idx = len(N_LEVEL_CLASSES)
        cath_levels = {k: torch.zeros(len(v), dtype=torch.float32) for k, v in self.cath_levels.items()}
        cath_levels.update({global_idx: torch.zeros(len(self.global_classes), dtype=torch.float32)})
        for cl in cath_cls:
            cath_levels[global_idx][self.global_classes.index(cl)] = 1.0
            for i, level in enumerate(cl.split('.')):
                cath_levels[i][self.cath_levels[i].index(int(level))] = 1.0

        return cath_levels

    def __len__(self):
        return len(self.items)


def get_loaders():
    cath_data = CATHData()
    cath_data.save_dataset()

    train_ds = CATHDataset()
    config = Config()
    test_ds = CATHDataset(mode='test')

    train_loader = DataLoader(train_ds,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)

    test_loader = DataLoader(test_ds,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=config.num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    train, _ = get_loaders()
    for batch in train:
        pass
