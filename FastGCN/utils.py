import torch
from torch_sparse import index_select
from torch_sparse.tensor import SparseTensor


def normalize_adj(adj, symmetric=True):
    """Convert adjacency matrix into normalized laplacian matrix

    Args:
        adj (torch sparse tensor): adjacency matrix
        symmetric (boolean) True: D^{-1/2}AD^{-1/2}; False: D^{-1}A

    Returns:
        (torch sparse tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=1).to_dense()
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

def sparse_norm(matrix, dim):
    """Sparse L2 norm, torch.norm currently only supports full reductions on sparse tensor

    Args:
        adj (torch sparse tensor): 2D matrix
        dim (int): dimension

    Returns:
        (torch tensor): norm vector
    """
    matrix = torch.sparse_coo_tensor(matrix._indices(), matrix._values().abs(), matrix.size()).pow(2)
    if dim == 0:
        norm = torch.sparse.sum(matrix, dim=1)._values().pow(0.5)
    if dim == 1:
        norm = torch.sparse.sum(matrix, dim=0)._values().pow(0.5)
    return norm

def sparse_select(adj, dim, index):
    """index select on sparse tensor (temporary function)
       torch.index_select on sparse tesnor is too slow to be useful
       https://github.com/pytorch/pytorch/issues/54561
    """
    adj = SparseTensor.from_torch_sparse_coo_tensor(adj)
    adj = index_select(adj, dim, index)
    return adj.to_torch_sparse_coo_tensor()
