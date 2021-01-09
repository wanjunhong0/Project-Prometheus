import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class Data(object):
    def __init__(self, path, test_size, seed):
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
        edge_list = torch.LongTensor(edge_list.values)
        # feature, label
        df = df.set_index(0)
        self.feature = torch.FloatTensor(df.iloc[:, :-1].values)
        self.feature = F.normalize(self.feature, p=1, dim=1)   # normalization
        # Pytorch require multi-class label as integer categorical class labels
        self.label = torch.LongTensor(np.where(pd.get_dummies(df.iloc[:, -1]).values)[1])
        self.n_class = df.iloc[:, -1].nunique()
        # graph
        self.n_node = df.shape[0]
        self.n_edge = edge_list.shape[0]

        adj = torch.sparse.FloatTensor(edge_list.T, torch.ones(self.n_edge), torch.Size([self.n_node, self.n_node]))
        # build symmetric adjacency matrix
        adj = torch.add(adj, adj.transpose(0, 1)) # may have elements > 1
        self.edge_list = adj._indices()
        self.neighbor_list = []
        for i in range(self.n_node):
            self.neighbor_list.append(self.edge_list[1][self.edge_list[0] == i])
        # train, val, test split
        self.idx_train, self.idx_test = train_test_split(range(self.n_node), test_size=test_size, random_state=seed)
        self.idx_train, self.idx_val = train_test_split(self.idx_train, test_size=test_size, random_state=seed)
        self.idx_train, self.idx_val, self.idx_test = [torch.LongTensor(i) for i in [self.idx_train, self.idx_val, self.idx_test]]


class Dataset(torch.utils.data.Dataset):
    """ Generate train, val, test dataset for the model """
    def __init__(self, neighbor_list, idx):
        self.neighbor_list = neighbor_list
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        """ Retrieve node_id and its 2-hop neighbor """
        idx = self.idx[i]
        neighbor = self.neighbor_list[idx]

        return neighbor, idx


class Collate(object):
    """ Collate function for mini-batch, can't use default collate_fn due to edge_list in different size"""
    def __init__(self, neighbor_list, n_sample, agg_type):
        self.n_sample = n_sample
        self.agg_type = agg_type
        self.neighbor_list = neighbor_list
        self.n_node = len(self.neighbor_list)

    def __call__(self, batch):
        """ Sample neighbors for each nodes in batch """
        neighbor = torch.cat([torch.LongTensor(i[0]) for i in batch])
        idx = torch.LongTensor([i[1] for i in batch])
        nodes = torch.unique(torch.cat([idx, neighbor]))
        edge_list = []
        for i in nodes:
            sample = self.neighbor_list[i]
            if len(sample) > self.n_sample and self.n_sample > 0:
                sample = sample[torch.randperm(len(sample))[: self.n_sample]]
            edge_list.append(torch.cat([torch.LongTensor([i] * len(sample)), sample]).view(2, -1))
        if self.agg_type == 'gcn':
            edge_list.append(torch.stack([nodes, nodes]))
        edge_list = torch.cat(edge_list, dim=1)
        adj = torch.sparse.FloatTensor(edge_list, torch.ones_like(edge_list[0]), torch.Size([self.n_node, self.n_node]))
        degree = torch.sparse.sum(adj, dim=1)._values()
        norm_adj = torch.sparse.FloatTensor(adj._indices(), torch.repeat_interleave(degree.float().pow(-1), degree), adj.size())

        return norm_adj, idx
