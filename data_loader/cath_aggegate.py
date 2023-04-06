import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from config.config import Config
from data_loader.cath_dataset import CATHData
from data_loader.geometric_features import ProteinFeatures


def aggregate_data_path():
    path_ = Path(__file__).parent.parent / 'dataset'
    aggregate_file = path_ / 'aggregate.pkl'
    return aggregate_file


class CATHAggregate(CATHData):
    def __init__(self):
        super().__init__()
        self.cath_aggregate = {}

        for cl in self.cath_cls:
            level3 = self.get_l3(cl)
            self.cath_aggregate.update({level3: []})

    def create_aggregate_set(self):
        for pdb_id in self.cath_data:
            cath_assign = self.cath_data[pdb_id]
            for cl in cath_assign:
                level3 = self.get_l3(cl)
                self.cath_aggregate[level3].append(pdb_id)

        dataset_path = aggregate_data_path()

        with open(dataset_path, 'wb') as fh:
            pickle.dump(self.cath_aggregate, fh)

    @staticmethod
    def get_l3(cath_rec):
        spl_ = cath_rec.split('.')
        level3 = f'{spl_[0]}.{spl_[1]}.{spl_[2]}'
        return level3


class CATHAggDataset(Dataset):
    def __init__(self):
        super().__init__()
        config = Config()
        dataset_path = aggregate_data_path()
        self.preprocessed_dir = Path(config.preprocessed_folder)
        assert os.path.exists(dataset_path)
        assert os.path.exists(self.preprocessed_dir)

        with open(dataset_path, 'rb') as fh:
            self.items = pickle.load(fh)

        self.cath_classes = list(self.items.keys())
        self.protein_features = ProteinFeatures()

    def __getitem__(self, index):
        sample = self.get_triplet(index)
        sample_features = []

        for file_key in sample:
            features = self.build_features(file_key)
            sample_features.append(features)

        return sample_features

    def build_features(self, file_key):
        filepath = self.preprocessed_dir / f'{file_key}.pkl'

        with open(filepath, 'rb') as fh:
            coo, sequence, _ = pickle.load(fh)

        features = self.protein_features.build_features((coo, sequence))

        return features

    def get_triplet(self, index):
        cath_classes = self.cath_classes.copy()
        positive_cls = cath_classes[index]
        positive_items = self.items[positive_cls].copy()
        anchor = np.random.choice(positive_items, 1)[0]

        if len(positive_items) > 1:
            positive_items.remove(anchor)
            positive = np.random.choice(positive_items, 1)[0]

        else:
            positive = anchor

        cath_classes.remove(positive_cls)
        negative_cls = np.random.choice(cath_classes)
        negative_items = self.items[negative_cls]
        negative = np.random.choice(negative_items, 1)[0]
        return anchor, positive, negative

    def __len__(self):
        return len(self.items)


def get_loaders():
    cath_data = CATHAggregate()
    cath_data.create_aggregate_set()

    train_ds = CATHAggDataset()
    config = Config()

    loader = DataLoader(train_ds,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers)

    return loader


if __name__ == '__main__':
    train, _ = get_loaders()
    for batch in train:
        pass
