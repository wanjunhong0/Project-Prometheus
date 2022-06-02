import torch
from utils import normalize_adj


class Sampler():
    def __init__(self, adj):
        self.adj = adj

    def DropEdge(self, rate):
        """Randomly drop edge"""
        nnz = self.adj._nnz()
        perm = torch.randperm(nnz)[:int(nnz*rate)]
        adj = torch.sparse_coo_tensor(self.adj._indices()[:, perm], self.adj._values()[perm], self.adj.size())

        return normalize_adj(adj, symmetric=True)
