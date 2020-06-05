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
        adj = (adj + adj.T.multiply(adj.T > adj)).todense()
        self.edge_dict = {}
        for i in range(adj.shape[0]):
            self.edge_dict[i] = set(np.where(adj[i] == 1.)[1])
        # self.norm_adj = create_norm_adj(adj, adj_type=adj_type)
        # train, val, test split
        self.idx_train, self.idx_test = train_test_split(range(self.n_node), test_size=test_size, random_state=seed)
        self.idx_train, self.idx_val = train_test_split(self.idx_train, test_size=test_size, random_state=seed)
        self.idx_train = torch.LongTensor(self.idx_train)
        self.idx_val = torch.LongTensor(self.idx_val)
        self.idx_test = torch.LongTensor(self.idx_test)
