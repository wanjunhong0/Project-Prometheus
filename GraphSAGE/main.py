import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from GraphSAGE.parser import parse_args
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

from GraphSAGE.encoders import Encoder
from GraphSAGE.aggregators import MeanAggregator
from GraphSAGE.models import SupervisedGraphSage
from GraphSAGE.load_data import Data

# Settings
args = parse_args()
torch.manual_seed(args.seed)

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset, adj_type=args.adj_type, test_size=args.test_size, seed=args.seed)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature
label = data.label
idx_train = data.idx_train
idx_val = data.idx_val
idx_test = data.idx_test
num_nodes = 2708
num_feats = 1433
feat_data = np.zeros((num_nodes, num_feats))


adj_lists = data.edge_dict
print(len(adj_lists))
"""
===========================================================================
Training
===========================================================================
"""

num_nodes = 2708
features = nn.Embedding(2708, 1433)
features.weight = nn.Parameter(feature, requires_grad=False)

agg1 = MeanAggregator(features)
# agg1_out = agg1(nodes, [self.adj_lists[int(node)] for node in nodes], 5)
enc1 = Encoder(features, 1433, 64, adj_lists, agg1, gcn=True, cuda=False)
# out1 = enc1()
agg2 = MeanAggregator(lambda nodes : enc1(nodes).t())
enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 64, adj_lists, agg2,
        base_model=enc1, gcn=True, cuda=False)
enc1.num_samples = 5
enc2.num_samples = 5

model = SupervisedGraphSage(7, enc2)


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(idx_train)
    loss_train = F.cross_entropy(input=output, target=label[idx_train])
    acc_train = accuracy_score(y_pred=output.max(1)[1], y_true=label[idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(idx_val)
    loss_val = F.cross_entropy(input=output, target=label[idx_val])
    acc_val = accuracy_score(y_pred=output.max(1)[1], y_true=label[idx_val])

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train.item() ,loss_val.item(), acc_train, acc_val))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(idx_test)
loss_test = F.cross_entropy(input=output, target=label[idx_test])
acc_test = accuracy_score(y_pred=output.max(1)[1], y_true=label[idx_test])
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test.item(), acc_test))
