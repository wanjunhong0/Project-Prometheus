import torch
from utils import normalize_adj, sparse_diag


class Sampler():
    """Uniform node sampling"""
    def __init__(self, adj, agg_type):
        self.adj = adj
        self.agg_type = agg_type
        self.n_node = self.adj.size()[0]
        self.neighbor_list = []
        for i in range(self.n_node):
            self.neighbor_list.append(self.adj[i]._indices()[0])

    def sampling(self, idx, n_sample):
        """Sampling adjacency matrix"""
        adj_layer2, idx_layer2 = self.node_sampling(idx, n_sample)
        adj_layer1, _ = self.node_sampling(idx_layer2, n_sample)

        return adj_layer1, adj_layer2

    def node_sampling(self, idx, n_sample):
        """Sampling neighbors per node"""
        edge = []
        for i in idx:
            sample = self.neighbor_list[i]
            n = len(sample)
            if 0 < n_sample < n:
                sample = sample[torch.randperm(n)[:n_sample]]
            edge.append(torch.stack([torch.LongTensor([i] * len(sample)), sample]))
        edge = torch.cat(edge, dim=1)
        adj = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), self.adj.size())
        if self.agg_type == 'gcn':
            adj = torch.add(adj, sparse_diag(torch.ones(self.n_node)))
        norm_adj = normalize_adj(adj, symmetric=False)
        idx = torch.unique(edge)

        return norm_adj, idx
