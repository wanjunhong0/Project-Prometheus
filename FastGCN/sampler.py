import torch
from utils import sparse_norm, sparse_diag, sparse_select
from torch_sparse import index_select
class Sampler():
    def __init__(self, feature, adj, n_sample):
        self.feature = feature
        self.adj = adj
        self.n_sample = n_sample
        norm = sparse_norm(adj)
        self.p = norm / torch.sum(norm)

    def sampling(self, idx):
        idx_layer2, adj_layer2 = self.layer_sampling(idx, self.n_sample)
        idx_layer1, adj_layer1 = self.layer_sampling(idx_layer2, self.n_sample)
        feature = self.feature[idx_layer1]

        return feature, adj_layer1, adj_layer2

    def layer_sampling(self, idx, n_sample):
        adj = sparse_select(self.adj, 0, idx)
        # adj = self.adj.index_select(0, idx)
        # neighbor = torch.sum(adj, dim=0).nonzero(as_tuple=False)
        neighbor = torch.unique(adj._indices()[1])
        p = self.p[neighbor]
        p = p / torch.sum(p)
        sample = torch.multinomial(p, n_sample, replacement=True)
        neighbor_sampled = neighbor[sample]
        p_sampled = p[sample]
        adj_sampled = torch.sparse.mm(sparse_select(adj, 1, sample), sparse_diag(1. / (p_sampled * n_sample)))

        return neighbor_sampled, adj_sampled
