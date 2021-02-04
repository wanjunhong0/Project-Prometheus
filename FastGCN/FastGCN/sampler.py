import torch


class Sampler():
    def __init__(self, feature, adj, n_sample):
        self.feature = feature
        self.adj = adj
        self.n_sample = n_sample
        self.p = self.sparse_norm(adj)

    def sparse_norm(self, adj):  # adj only contains 1, no need to pow(2)
        degree = torch.sparse.sum(adj, dim=1)._values()
        return degree.sqrt() / degree

    def sampling(self, idx):
        idx_layer2, adj_layer2 = self.layer_sampling(idx, self.n_sample)
        idx_layer1, adj_layer1 = self.layer_sampling(idx_layer2, self.n_sample)
        feature = self.feature[idx_layer1]

        return feature, adj_layer1, adj_layer2

    def layer_sampling(self, idx, n_sample):
        adj = self.adj.index_select(0, idx)
        # neighbor = torch.sum(adj, dim=0).nonzero(as_tuple=False)
        neighbor = torch.unique(adj._indices()[1])
        p = self.p[neighbor]
        p = p / torch.sum(p)
        sample = torch.multinomial(p, n_sample, replacement=True)
        neighbor_sampled = neighbor[sample]
        p_sampled = p[sample]
        adj_sampled = torch.sparse.mm(adj.index_select(1, sample), torch.diag(1. / (p_sampled * n_sample)))

        return neighbor_sampled, adj_sampled
