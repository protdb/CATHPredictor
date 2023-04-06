import math

import numpy as np
import torch
import torch_cluster
import torch_geometric
from Bio.PDB.Polypeptide import three_to_index
import torch.nn.functional as F
from model.model_utils import normalize, ModelParams


class ProteinFeatures(object):
    def __init__(self):
        config = ModelParams()
        self.top_k = config.neighbours_agg
        self.num_positional_embeddings = config.num_positional_embeddings
        self.alphabet_size = config.alphabet_size
        self.coo = None

    def build_features(self, record):
        coo, sequence = record
        self.coo = torch.from_numpy(coo)
        mask = torch.isfinite(self.coo.sum(dim=(1, 2)))
        self.coo[~mask] = np.inf
        sequence_idx = self.sequence_to_index(sequence)
        with torch.no_grad():
            node_scalar_features, node_vector_features = self.get_node_features()
            edge_index, edge_scalar_features, edge_vector_features = self.get_edge_features()

        data = torch_geometric.data.Data(x=self.coo[:, 1, :],
                                         sequence=sequence_idx,
                                         node_features=(node_scalar_features, node_vector_features),
                                         edge_scalar_features=edge_scalar_features,
                                         edge_vector_features=edge_vector_features,
                                         edge_index=edge_index)
        return data

    def get_node_features(self):
        ca_coo = self.coo[:, 1, :]
        dihedrals = self._dihedrals(self.coo)
        orientations = self._orientations(ca_coo)
        sidechains = self._sidechains(self.coo)
        node_scalar_features = dihedrals
        node_vector_features = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        node_scalar_features, node_vector_features = map(torch.nan_to_num, (node_scalar_features, node_vector_features))

        return node_scalar_features, node_vector_features

    def get_edge_features(self):
        ca_coo = self.coo[:, 1]
        edge_index = torch_cluster.knn_graph(ca_coo, k=self.top_k)
        pos_embeddings = self._positional_embeddings(edge_index)
        e_vectors = ca_coo[edge_index[0]] - ca_coo[edge_index[1]]
        edge_vector_features = normalize(e_vectors).unsqueeze(-2)
        dist = e_vectors.norm(dim=-1)
        dist = torch.nan_to_num(dist)
        pos_embeddings = torch.nan_to_num(pos_embeddings)
        edge_vector_features = torch.nan_to_num(edge_vector_features)
        edge_scalar_features = (dist, pos_embeddings)

        return edge_index, edge_scalar_features, edge_vector_features

    def sequence_to_index(self, sequence):
        sequence_index = []
        for items in sequence:
            try:
                idx = three_to_index(items)
            except KeyError:
                idx = self.alphabet_size - 1
            assert 0 <= idx < self.alphabet_size
            sequence_index.append(idx)
        return torch.as_tensor(sequence_index, dtype=torch.long)

    @staticmethod
    def _orientations(x):
        forward = normalize(x[1:] - x[:-1])
        backward = normalize(x[:-1] - x[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    @staticmethod
    def _dihedrals(x, eps=1e-7):
        x = torch.reshape(x[:, :3], [3 * x.shape[0], 3])
        dX = x[1:] - x[:-1]
        U = normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]
        n_2 = normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = normalize(torch.cross(u_1, u_0), dim=-1)
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    @staticmethod
    def _sidechains(X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)

        return vec

    def _positional_embeddings(self, edge_index):
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, self.num_positional_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_positional_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
