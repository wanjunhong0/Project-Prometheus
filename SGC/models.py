import torch
import torch.nn.functional as F


class SGC(torch.nn.Module):
    def __init__(self, n_feature, n_class, dropout):
        """
        Args:
            n_feature (int): the dimension of feature
            n_class (int): the number of classification label
            dropout (float): dropout rate
        """
        super(SGC, self).__init__()

        self.dropout = dropout
        self.fc = torch.nn.Linear(n_feature, n_class)

    def forward(self, feature):
        """
        Args:
            feature (torch Tensor): feature input

        Returns:
            (torch Tensor): log probability for each class in label
        """
        x = self.fc(feature)
        x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)
