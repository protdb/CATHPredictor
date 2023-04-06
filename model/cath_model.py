import os.path
from pathlib import Path

import torch
import torch.nn as nn

from config.config import Config
from model.base_model import ABBModel
from model.hierarchy_pool import HSPool


class CATHPredictorModel(nn.Module):
    def __init__(self, load_base_model=True):
        super().__init__()
        self.base_model = ABBModel().float()
        if load_base_model:
            self.load_base_model()
        self.hsp_model = HSPool().float()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='sum')

    def load_base_model(self):
        config = Config()
        model_path = config.base_model_path

        if os.path.exists(model_path):
            print(f'Base model loaded {model_path}')
            self.base_model.load_state_dict(torch.load(model_path), strict=False)
        else:
            print(f'Base model not found {model_path}')

    def forward(self, sample):
        sample_input = {}

        for idx, batch in enumerate(sample):
            node_embeddings = self.base_out(batch, augment=idx == 1)
            sample_input.update({idx: (node_embeddings, batch)})

        triplets = []

        for idx in sample_input:
            node_embeddings, batch = sample_input[idx]
            embedding = self.hsp_model(node_embeddings, batch.edge_index, batch.batch)
            triplets.append(embedding)

        triplet_loss = self.triplet_loss(*triplets)
        return triplet_loss

    @torch.no_grad()
    def get_embeddings(self, batch):
        v1, v2, saml = self.base_model(batch)
        node_embeddings = torch.cat((v1, v2), dim=-1)
        embedding = self.hsp_model(node_embeddings, batch.edge_index, batch.batch)
        return embedding, saml

    @torch.no_grad()
    def base_out(self, batch, augment=False):
        if augment:
            self.base_model.train()
        else:
            self.base_model.eval()
        v1, v2, _ = self.base_model(batch)
        out = torch.cat((v1, v2), dim=-1)
        return out
