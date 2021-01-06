import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, in_dim, out_dim, agg_type): 
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            agg_type (str): the type of Aggregator
        """
        super(Aggregator, self).__init__()

        self.W = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.agg_type = agg_type

        torch.nn.init.xavier_uniform_(self.W)


    def forward(self, input, adj):
        """
        Args:
            input (torch Tensor): H hiddens
            adj (torch Tensor): normalized adjacency matrix

        Returns:
            (torch Tensor): H after aggregation
        """
        h = torch.sparse.mm(adj, input)
        output = torch.mm(torch.cat([h, input], dim=1), self.W)

        return F.relu(output)

        
