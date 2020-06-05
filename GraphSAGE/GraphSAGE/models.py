import torch
from torch.nn.parameter import Parameter
from GraphSAGE.encoders import Encoder
from GraphSAGE.aggregators import MeanAggregator


class SupervisedGraphSage(torch.nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc


        self.fc = torch.nn.Linear(64, num_classes, bias=True)

    def forward(self, nodes):
        embeds = self.enc(nodes).t()
        scores = self.fc(embeds)
        return scores



