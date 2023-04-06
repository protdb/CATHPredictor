import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

PAD_SEQ = 'Z'
SAML = ['A', 'Y', 'B', 'C', 'D', 'G', 'I', 'L', 'E', 'F', 'H', 'K', 'N', 'S', 'T', 'V', 'W', 'X', 'M', 'P', 'Q', 'R',
        PAD_SEQ]

STRIDE_LETTERS = ['H', 'G', 'I', 'E', 'B', 'b', 'T', 'C', ' ']


class ModelParams(object):
    alphabet_size = 21
    sequence_emb_dim = 21
    num_rbf = 16
    neighbours_agg = 32
    num_positional_embeddings = 16
    dihedral_embed_dim = 16
    rbf_cutoff_lower = 0.0
    rbf_cutoff_upper = 24.0
    hidden_emb_nodes = 64
    n_gvp_encoder_layers = 3
    drop_rate = 0.3
    rnn_dim = 32
    negative_from_anchor_factor = 0.3
    pad = 2048
    debug = False


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class GaussianSmearing(nn.Module):
    def __init__(self):
        super(GaussianSmearing, self).__init__()
        config = ModelParams()
        self.cutoff_lower = config.rbf_cutoff_lower
        self.cutoff_upper = config.rbf_cutoff_upper
        self.num_rbf = config.num_rbf
        offset, coeff = self._initial_params()
        self.register_parameter("coeff", nn.Parameter(coeff))
        self.register_parameter("offset", nn.Parameter(offset))

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class DihedralFeatures(nn.Module):
    # 3 dihedral angles; sin and cos of each angle
    node_in = 6

    def __init__(self):
        super(DihedralFeatures, self).__init__()
        config = ModelParams()
        node_embed_dim = config.dihedral_embed_dim
        self.node_embedding = nn.Linear(self.node_in, node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, x):
        v = self.node_embedding(x)
        v = self.norm_nodes(v)
        return v


class ReadoutModule(torch.nn.Module):
    def __init__(self, hdim):
        super(ReadoutModule, self).__init__()

        self.weight = torch.nn.Parameter(torch.Tensor(hdim, hdim))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return global_add_pool(weighted, batch)


class Set2Set(torch.nn.Module):
    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

def normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)
