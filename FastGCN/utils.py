import torch


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

def sparse_norm(adj):  # adj only contains 1, no need to pow(2)
    degree = torch.sparse.sum(adj, dim=1)._values()
    return degree.sqrt() / degree

def sparse_select(adj, dim, index):
    adj = adj.to_dense()
    if dim == 0:
        adj = adj[index]
    if dim == 1:
        adj = adj[:, index]
    return adj.to_sparse()
