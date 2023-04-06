import math

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv

from config.config import EncoderConfig
from model.model_utils import Set2Set, ReadoutModule


class HSPool(nn.Module):
    def __init__(self):
        super(HSPool, self).__init__()
        encoder_cfg = EncoderConfig()
        self.global_pool = Set2Set(encoder_cfg.input_dim, processing_steps=64, num_layers=2)
        self.n_layers = encoder_cfg.n_layers
        self.h_pool = nn.ModuleList(
            HLocalPool(input_dim=encoder_cfg.input_dim, glob_hdim=encoder_cfg.input_dim * 2)
            for _ in range(encoder_cfg.n_layers)
        )
        self.ff = nn.Sequential(
            nn.Linear(5 * encoder_cfg.input_dim, 4 * encoder_cfg.input_dim),
            nn.SiLU(),
            nn.Linear(4 * encoder_cfg.input_dim, 3 * encoder_cfg.input_dim)
        )

    def forward(self, x, edge_index, batch):
        global_x = self.global_pool(x, batch)
        output = []

        for i in range(self.n_layers):
            x, out = self.h_pool[i](x, edge_index, batch, global_x)
            output.append(out)

        output = torch.cat(output, dim=-1)
        output = torch.cat((output, global_x), dim=-1)
        output = self.ff(output)
        return output


class HLocalPool(torch.nn.Module):
    def __init__(self, input_dim, glob_hdim, num_heads=4):
        super(HLocalPool, self).__init__()
        self.gat = GATConv(in_channels=input_dim, out_channels=input_dim, num_heads=num_heads)
        self.read_out = ReadoutModule(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim + glob_hdim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x, edge_index, batch, x_global):
        x_att = self.gat(x, edge_index)
        pool = self.read_out(x_att, batch)
        x_local = torch.cat((x_global, pool), dim=-1)
        x_local = self.ff(x_local)
        return x_att, x_local
