import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class Data(object):
    def __init__(self, path):
        self.path = path
        # Reading files
        df = pd.read_csv('./data/cora/cora.content', header=None, sep='\t')
        edge_list = pd.read_csv('./data/cora/cora.cites', header=None, sep='\t')
        # map index starting from 0 to n
        id_map = dict(zip(df[0], range(df.shape[0])))
        df[0] = df[0].map(id_map)
        edge_list[0] = edge_list[0].map(id_map)
        edge_list[1] = edge_list[1].map(id_map)
        # feature, label
        df = df.set_index(0)
        self.feature = torch.FloatTensor(df.iloc[:, :-1].values)
        self.feature = F.normalize(self.feature, p=1, dim=1)   #normalization
        # self.label = torch.LongTensor(pd.get_dummies(df.iloc[:, -1]).values)
        self.label = torch.LongTensor(np.where(pd.get_dummies(df.iloc[:, -1]).values)[1])
        # graph
        n_node = df.shape[0]
        n_edge = edge_list.shape[0]
        adj = sp.coo_matrix((np.ones(n_edge), (edge_list[0], edge_list[1])), shape=(n_node, n_node),dtype=np.float32)
















