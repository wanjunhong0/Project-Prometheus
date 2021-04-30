import torch


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

    return torch.sparse_coo_tensor(index, vector._values(), [n ,n])
