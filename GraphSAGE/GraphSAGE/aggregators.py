import torch
from torch.autograd import Variable
import random
import numpy as np
import scipy.sparse as sp

"""
Set of modules for aggregating embeddings of neighbors.
"""
class MeanAggregator(torch.nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, n_sample): 
        """
        Args:
            n_sample (int): the number of neighbor that sampled for each node. No sampling if None.
        """
        self.n_sample = n_sample
        super(MeanAggregator, self).__init__()
        
    def forward(self, feature, neighbor_list):
        """
        Args:
            feature (torch Tensor): feature input
            neighbor_list (numpy array): neighbor id list for each node

        Returns:
            (torch Tensor): feature after mean aggregation
        """
        if self.n_sample is None:
            sample = neighbor_list
        else:
            sample = [set(random.sample(neighbor, self.n_sample)) if len(neighbor) >= self.n_sample 
                      else neighbor for neighbor in neighbor_list]

        # not likely all nodes will be sampled, only aggregate those sampled nodes to reduce time consumption
        unique_nodes_list = list(set.union(*sample))
        unique_nodes_dict = {node_id: i for i, node_id in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(sample), len(unique_nodes_dict)))
        column_indices = [unique_nodes_dict[node_id] for sample_neighbor in sample for node_id in sample_neighbor]   
        row_indices = [i for i in range(len(sample)) for node_id in range(len(sample[i]))]
        mask[row_indices, column_indices] = 1
        n_neighbor = mask.sum(1, keepdim=True)
        mask = mask.div(n_neighbor)
        # convert sparse multiplication to reduce time consumption
        indices = torch.nonzero(mask).t()
        mask = torch.sparse.FloatTensor(indices, mask[indices[0], indices[1]], mask.size())

        # agg_feature = mask.mm(feature[torch.LongTensor(unique_nodes_list)])
        agg_feature = torch.spmm(mask, feature[torch.LongTensor(unique_nodes_list)])
        return agg_feature
