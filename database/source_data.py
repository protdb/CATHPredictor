import pickle

from torch_geometric.loader import DataLoader
from config.config import Config
from data_loader.cath_aggegate import CATHAggDataset


class SourceDataset(CATHAggDataset):
    def __init__(self):
        super().__init__()

        self.data = {}
        files_ids = {}

        for cath_id in self.items:
            files = self.items[cath_id]
            for file_id in files:
                if file_id in files_ids:
                    files_ids[file_id].append(cath_id)
                else:
                    files_ids.update({file_id: [cath_id]})

        for idx, (k, v) in enumerate(files_ids.items()):
            self.data.update({idx: {k: v}})

    def __getitem__(self, item):
        record = self.data[item]
        file_id = list(record.keys())[0]
        features = self.build_features(file_id)
        features.data_idx = item

        return features

    def save_indices(self):
        config = Config()

        with open(config.index_file, 'wb') as fh:
            pickle.dump(self.data, fh)

    def __len__(self):
        return len(self.data)


def get_source_loader():
    source_ds = SourceDataset()
    source_ds.save_indices()
    config = Config()
    loader_ = DataLoader(source_ds,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    return loader_
