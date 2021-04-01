import torch
from torch_geometric.datasets import Planetoid


class Data():
    def __init__(self, path, dataset):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            adj_type (str): the type of laplacian matrix of adjacency matrix {'single', 'double'}
            test_size (float): the testset proportion of the dataset
            seed (int): random seed
        """
        # load data
        data = Planetoid(root=path, name=dataset)
        self.feature = data[0].x
        self.edge = data[0].edge_index
        self.label = data[0].y
        self.mask_train = data[0].train_mask
        self.mask_val = data[0].val_mask
        self.mask_test = data[0].test_mask
        self.n_node = data[0].num_nodes
        self.n_edge = data[0].num_edges
        self.n_class = data.num_classes
        self.n_feature = data.num_features
        # Calculate adj
        self.adj = torch.sparse_coo_tensor(self.edge, torch.ones(self.n_edge), [self.n_node, self.n_node])
        self.norm_adj = normalize_adj(torch.add(self.adj, sparse_diag(torch.ones(self.n_node))), symmetric=True)


def normalize_adj(adj, symmetric):
    """Convert adjacency matrix into normalized laplacian matrix

    Args:
        adj (torch sparse tensor): adjacency matrix
        symmetric (boolean) True: D^{-1/2}AD^{-1/2}; False: D^{-1}A

    Returns:
        (torch sparse tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=1)._values()
    if symmetric:
        degree_ = sparse_diag(degree.pow(-0.5))
        norm_adj = torch.sparse.mm(torch.sparse.mm(degree_, adj), degree_)
    else:
        degree_ = sparse_diag(degree.pow(-1))
        norm_adj = torch.sparse.mm(degree_, adj)

    return norm_adj


def sparse_diag(value):
    """Convert vector into diagonal matrix

    Args:
        value (torch tensor): vector

    Returns:
        (torch sparse tensor): sparse matrix with only diagonal values
    """
    n = len(value)
    index = torch.stack([torch.arange(n), torch.arange(n)])

    return torch.sparse_coo_tensor(index, value, [n ,n])
