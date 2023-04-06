import torch
import torch.nn as nn
from model.gvp_layers import GVP, LayerNorm, GVPConvLayer
from model.model_utils import GaussianSmearing, DihedralFeatures, ModelParams, SAML, STRIDE_LETTERS
import torch_geometric


class ABBModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = ModelParams()
        self._gaussian_smearing = GaussianSmearing()
        self._dihedral_features = DihedralFeatures()
        self.embed_seq = nn.Embedding(config.alphabet_size, config.alphabet_size)
        input_emb_nodes_dim = (config.dihedral_embed_dim, 3)
        hidden_emb_nodes_dim = config.hidden_emb_nodes
        input_emb_edges_dim = (config.num_positional_embeddings + config.num_rbf, 1)
        hidden_emb_edges_dim = input_emb_edges_dim[0] * 2
        hidden_emb_nodes_dim = (hidden_emb_nodes_dim, hidden_emb_nodes_dim)
        hidden_emb_edges_dim = (hidden_emb_edges_dim, hidden_emb_edges_dim)
        self.gvp_embed_nodes = nn.Sequential(
            GVP(input_emb_nodes_dim, hidden_emb_nodes_dim, activations=(None, None)),
            LayerNorm(hidden_emb_nodes_dim)
        )
        self.gvp_embed_edges = nn.Sequential(
            GVP(input_emb_edges_dim, hidden_emb_edges_dim, activations=(None, None)),
            LayerNorm(hidden_emb_edges_dim)
        )
        hidden_emb_edges_dim = (hidden_emb_edges_dim[0] + config.alphabet_size, hidden_emb_edges_dim[1])

        self.gvp_encoder = nn.ModuleList(
            GVPConvLayer(hidden_emb_nodes_dim,
                         hidden_emb_edges_dim,
                         drop_rate=config.drop_rate
                         )
            for _ in range(config.n_gvp_encoder_layers))

        h_node_out_emb_dim = (hidden_emb_nodes_dim[0] * 2, hidden_emb_nodes_dim[1])
        self.gvp_decoder = nn.ModuleList(
            GVPConvLayer(h_node_out_emb_dim,
                         hidden_emb_edges_dim,
                         drop_rate=config.drop_rate,
                         autoregressive=True
                         )
            for _ in range(config.n_gvp_encoder_layers))

        self.gvp_out = nn.Sequential(
            LayerNorm(h_node_out_emb_dim),
            GVP(h_node_out_emb_dim, (len(SAML) + 1, 0),
                activations=(None, None))
        )

        self.gvp_out_stride = nn.Sequential(
            LayerNorm(h_node_out_emb_dim),
            GVP(h_node_out_emb_dim, (len(STRIDE_LETTERS), 0),
                activations=(None, None))
        )

        self.encoder_rnn = nn.LSTM(hidden_emb_nodes_dim[0],
                                   hidden_emb_nodes_dim[0],
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True,
                                   bias=True)

        self.decoder_rnn = nn.ModuleList(nn.LSTM(hidden_emb_nodes_dim[0] * 2,
                                                 hidden_emb_nodes_dim[0],
                                                 bidirectional=True,
                                                 batch_first=True,
                                                 bias=True)
                                         for _ in range(config.n_gvp_encoder_layers))

    def forward(self, batch):
        h_node_embeddings, h_edge_embeddings = self.build_features_embeddings(batch)

        for i, layer in enumerate(self.gvp_encoder):
            h_node_embeddings = layer(h_node_embeddings, batch.edge_index, h_edge_embeddings)

        rnn_embeddings, rnn_mask = torch_geometric.utils.to_dense_batch(h_node_embeddings[0], batch.batch)
        rnn_embeddings, _ = self.encoder_rnn(rnn_embeddings)
        rnn_embeddings = rnn_embeddings[rnn_mask]
        h_node_embeddings = (rnn_embeddings, h_node_embeddings[1])
        encoder_embeddings = h_node_embeddings

        for i, layer in enumerate(self.gvp_decoder):
            h_node_embeddings = layer(h_node_embeddings, batch.edge_index, h_edge_embeddings,
                                      autoregressive_x=encoder_embeddings)
            rnn_embeddings, rnn_mask = torch_geometric.utils.to_dense_batch(h_node_embeddings[0], batch.batch)
            rnn_embeddings, _ = self.decoder_rnn[i](rnn_embeddings)
            rnn_embeddings = rnn_embeddings[rnn_mask]
            h_node_embeddings = (rnn_embeddings, h_node_embeddings[1])

        saml_ = self.gvp_out(h_node_embeddings)

        return encoder_embeddings[0], h_node_embeddings[0], saml_

    def build_features_embeddings(self, batch):
        node_scalar_features, node_vector_features = batch.node_features
        node_scalar_features = self._dihedral_features(node_scalar_features)
        embed_seq = self.embed_seq(batch.sequence)
        dist, pos_embeddings = batch.edge_scalar_features
        rbf = self._gaussian_smearing(dist)
        edge_scalar_embeddings = torch.cat((rbf, pos_embeddings), dim=-1)
        edge_vector_embeddings = batch.edge_vector_features
        h_node_embeddings = self.gvp_embed_nodes((node_scalar_features, node_vector_features))
        h_edge_embeddings = self.gvp_embed_edges((edge_scalar_embeddings, edge_vector_embeddings))
        embed_seq = embed_seq[batch.edge_index[0]]
        embed_seq[batch.edge_index[0] >= batch.edge_index[1]] = 0
        h_edge_embeddings = (torch.cat([h_edge_embeddings[0], embed_seq], dim=-1), h_edge_embeddings[1])
        return h_node_embeddings, h_edge_embeddings
