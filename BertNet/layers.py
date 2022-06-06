import torch
from scipy.special import comb
import torch.nn.functional as F
from torch.nn import Parameter


class Bert_Propagation(torch.nn.Module):
    def __init__(self, k):
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            k (int): k -th propagation
        """
        super(Bert_Propagation, self).__init__()

        self.k = k
        self.temp = Parameter(torch.Tensor(k+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, input, adj, adj_):
        """
        Args:
            input (torch tensor): H Hiddens
            adj (torch tensor): L Laplacian matrix
            adj (torch tensor): 2I-L Laplacian matrix
        """
        temp = F.relu(self.temp)
        x = input
        xs = [x]
        for _ in range(self.k):
            xs.append(torch.sparse.mm(adj_, x))
        output = (comb(self.k, 0) / (2 ** self.k)) * temp[0] * xs[self.k]
        for i in range(self.k):
            x = torch.sparse.mm(adj, xs[self.k-i-1])
            for _ in range(i):
                x = torch.sparse.mm(adj, x)
        output = output + (comb(self.k, i + 1) / (2 ** self.k)) * temp[i+1] * x

        return output
