import torch
import torch.nn.functional as F


class DAGNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, k, dropout):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            k (int): k-hop aggregation
            dropout (float): dropout rate
        """
        super(DAGNN, self).__init__()

        self.k = k
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(n_feature, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, n_class)
        self.alpha = torch.nn.Linear(n_class, 1)

    def forward(self, feature, adj):
        """
        Args:
            feature (torch Tensor): feature input
            adj (torch Tensor): Laplacian matrix

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = F.dropout(feature, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        xs = [x]
        for _ in range(self.k):
            x = torch.sparse.mm(adj, x)
            xs.append(x)
        xs = torch.stack(xs, dim=1)
        score = torch.sigmoid(self.alpha(xs).squeeze()).unsqueeze(1)
        out = torch.matmul(score, xs).squeeze()

        return F.log_softmax(out, dim=1)
