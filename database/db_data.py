import pickle

import numpy as np
import faiss

from config.config import Config, EncoderConfig
from utils.utils import convert_to_sa


class DatabaseData(object):
    def __init__(self):
        self.sa_data = {}
        self.embeddings_data = {}
        self.config = Config()
        encoder_cfg = EncoderConfig()
        self.embeddings_dim = encoder_cfg.h_dim * encoder_cfg.n_layers

    def append_batch(self,
                     embeddings,
                     saml_,
                     data_indices,
                     batch_ids,
                     ):
        self.append_saml(saml_, data_indices, batch_ids)
        self.append_embeddings(embeddings, data_indices)

    def append_saml(self, saml_, indices, batch_ids):
        batch_size = np.max(batch_ids) + 1

        for i in range(batch_size):
            mask = batch_ids == i
            sa_ids = saml_[mask]
            sa = convert_to_sa(sa_ids)
            data_idx = indices[i]
            self.sa_data.update({data_idx: sa})

    def append_embeddings(self, embeddings, indices):
        for i, idx in enumerate(indices):
            self.embeddings_data.update({idx: embeddings[i, :]})

    def __save_embeddings_data(self):
        quantizer = faiss.IndexFlatL2(self.embeddings_dim)
        fs_index = faiss.IndexIDMap(quantizer)
        vectors = np.vstack(list(self.embeddings_data.values()))
        indices = np.array(list(self.embeddings_data.keys()))
        fs_index.add_with_ids(vectors, indices)
        faiss.write_index(fs_index, str(self.config.data_file))

    def __save_sa_data(self):
        with open(self.config.sa_data_file, 'wb') as fh:
            pickle.dump(self.sa_data, fh)

    def save_data(self):
        self.__save_embeddings_data()
        self.__save_sa_data()
