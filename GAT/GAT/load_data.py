import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import argparse


class Data(object):
    def __init__(self, path, adj_type, test_size, seed):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index
           
        Args:
            path (str): file path
            adj_type (str): the type of laplacian matrix of adjacency matrix {'single', 'double'}
            test_size (float): the testset proportion of the dataset 
            seed (int): random seed
        """
        # Reading files
        df = pd.read_csv(path + '/cora.content', header=None, sep='\t')
        edge_list = pd.read_csv(path + '/cora.cites', header=None, sep='\t')
        # map index starting from 0 to n
        id_map = dict(zip(df[0], range(df.shape[0])))
        df[0] = df[0].map(id_map)
        edge_list[0] = edge_list[0].map(id_map)
        edge_list[1] = edge_list[1].map(id_map)
        # feature, label
        df = df.set_index(0)
        self.feature = torch.FloatTensor(df.iloc[:, :-1].values)
        self.feature = F.normalize(self.feature, p=1, dim=1)   #normalization
        # Pytorch require multi-class label as integer categorical class labels
        self.label = torch.LongTensor(np.where(pd.get_dummies(df.iloc[:, -1]).values)[1])
        self.n_class = df.iloc[:, -1].nunique()
        # graph
        self.n_node = df.shape[0]
        self.n_edge = edge_list.shape[0]
        adj = sp.coo_matrix((np.ones(self.n_edge), (edge_list[0], edge_list[1])), shape=(self.n_node, self.n_node),dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj)
        self.norm_adj = create_norm_adj(adj, adj_type=adj_type)
        self.norm_adj = torch.FloatTensor(np.array(self.norm_adj.todense()))
        # train, val, test split
        self.idx_train, self.idx_test = train_test_split(range(self.n_node), test_size=test_size, random_state=seed)
        self.idx_train, self.idx_val = train_test_split(self.idx_train, test_size=test_size, random_state=seed)
        self.idx_train = torch.LongTensor(self.idx_train)
        self.idx_val = torch.LongTensor(self.idx_val)
        self.idx_test = torch.LongTensor(self.idx_test)


def create_norm_adj(adj_mat, adj_type):
    """Create normalized laplacian matrix from adjacency matrix

    Args:
        adj_mat (scipy.sparse matrix): adjacency matrix
        adj_type (str): the type of laplacian matrix of adjacency matrix {'single', 'double'}

    Returns:
        (torch.sparse Tensor): Normalized laplacian matrix
    """
    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))
        with np.errstate(divide='ignore'):   # ignore divide by zero warning
            d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        print('Using Random walk normalized Laplacian.')
        return norm_adj.tocoo()

    def normalized_adj_double(adj):
        rowsum = np.array(adj.sum(1))
        with np.errstate(divide='ignore'):  # ignore divide by zero warning
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = (d_mat_inv_sqrt.dot(adj)).dot(d_mat_inv_sqrt)
        print('Using Symmetric normalized Laplacian')
        return norm_adj.tocoo()
    
    if adj_type == 'single':     # D^(-1)(A)
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    if adj_type == 'double':    # D^(-1/2)AD^(-1/2)
        norm_adj_mat = normalized_adj_double(adj_mat + sp.eye(adj_mat.shape[0]))

    return norm_adj_mat
    # return sparse_mx_to_torch_sparse_tensor(norm_adj_mat)


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)















