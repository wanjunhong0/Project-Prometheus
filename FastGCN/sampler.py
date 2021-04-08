import torch
from utils import sparse_norm, sparse_diag, sparse_select


class Sampler():
    def __init__(self, feature, adj, n_sample):
        """Importance sampling on graph
           https://arxiv.org/abs/1801.10247
        """
        self.feature = feature
        self.adj = adj
        self.n_sample = n_sample
        norm = sparse_norm(adj, dim=0)
        self.p = norm / torch.sum(norm)

    def sampling(self, idx):
        """Sampling adjacency matrix for each layer for graph convolution"""
        idx_layer2, adj_layer2 = self.layer_sampling(idx, self.n_sample)
        idx_layer1, adj_layer1 = self.layer_sampling(idx_layer2, self.n_sample)
        feature = self.feature[idx_layer1]

        return feature, adj_layer1, adj_layer2

    def layer_sampling(self, idx, n_sample):
        """Importance sampling based on norm of the normailized adjacency matrix as probability"""
        adj = sparse_select(self.adj, 0, idx)
        neighbor = torch.sparse.sum(adj, dim=0)._indices()[0]
        # neighbor = torch.unique(adj._indices()[1])
        p = self.p[neighbor]
        p = p / torch.sum(p)
        sample = torch.multinomial(p, n_sample, replacement=True)
        neighbor_sampled = neighbor[sample]
        p_sampled = p[sample]
        adj_sampled = torch.sparse.mm(sparse_select(adj, 1, neighbor_sampled), sparse_diag(1. / (p_sampled * n_sample)))

        return neighbor_sampled, adj_sampled
