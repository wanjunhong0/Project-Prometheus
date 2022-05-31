import torch


def normalize_adj(adj, symmetric=True):
    """Convert adjacency matrix into normalized laplacian matrix

    Args:
        adj (torch sparse tensor): adjacency matrix
        symmetric (boolean) True: D^{-1/2}AD^{-1/2}; False: D^{-1}A

    Returns:
        (torch sparse tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=1)
    if symmetric:
        degree_ = sparse_diag(degree.pow(-0.5))
        norm_adj = torch.sparse.mm(torch.sparse.mm(degree_, adj), degree_)
    else:
        degree_ = sparse_diag(degree.pow(-1))
        norm_adj = torch.sparse.mm(degree_, adj)

    return norm_adj


def sparse_diag(vector):
    """Convert vector into diagonal matrix

    Args:
        vector (torch tensor): diagonal values of the matrix

    Returns:
        (torch sparse tensor): sparse matrix with only diagonal values
    """
    if not vector.is_sparse:
        vector = vector.to_sparse()
    n = len(vector)
    index = torch.stack([vector._indices()[0], vector._indices()[0]])

    return torch.sparse_coo_tensor(index, vector._values(), [n, n])


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
