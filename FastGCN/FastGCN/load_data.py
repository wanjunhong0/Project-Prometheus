import pandas as pd
import torch
import torch.nn.functional as F


class Data():
    def __init__(self, path, adj_type, val_test):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            adj_type (str): the type of laplacian matrix of adjacency matrix {'single', 'double'}
            val_test (list): the val and test set proportion of the dataset
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
        self.label = torch.argmax(torch.LongTensor(pd.get_dummies(df.iloc[:, -1]).values), dim=1)
        self.n_class = df.iloc[:, -1].nunique()
        # graph
        self.n_node = df.shape[0]
        self.n_edge = edge_list.shape[0]
        adj = torch.sparse.FloatTensor(edge_list.T, torch.ones(self.n_edge), torch.Size([self.n_node, self.n_node]))
        # build symmetric adjacency matrix
        adj = torch.add(torch.eye(self.n_node).to_sparse(), torch.add(adj, adj.transpose(0, 1))) # may have elements > 1
        self.adj = torch.sparse.FloatTensor(adj._indices(), torch.ones_like(adj._values()), adj.size())
        self.norm_adj = create_norm_adj(adj, adj_type=adj_type)
        # train, val, test split
        val_test_size = [int(self.n_node * i) for i in val_test]
        split_size = [self.n_node - sum(val_test_size)] + val_test_size
        self.idx_train, self.idx_val, self.idx_test = torch.randperm(self.n_node).split(split_size)



class Dataset(torch.utils.data.Dataset):
    """ Generate train, val, test dataset for the model """
    def __init__(self, idx):
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i]

        return idx


def create_norm_adj(adj, adj_type):
    """Create normalized laplacian matrix from adjacency matrix

    Args:
        adj_mat (troch sparse tensor): adjacency matrix
        adj_type (str): the type of laplacian matrix of adjacency matrix

    Returns:
        (torch.sparse Tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=1)._values()
    if adj_type == 'unsymmetric':  # D^(-1) * A
        degree_ = torch.diag(degree.pow(-1))
        norm_adj = torch.mm(adj, degree_)
    if adj_type == 'symmetric':   # D^(-1/2) * A * D^(-1/2)
        degree_ = torch.diag(degree.pow(-0.5))
        norm_adj = torch.mm(torch.mm(adj, degree_), degree_)

    return norm_adj.to_sparse()
