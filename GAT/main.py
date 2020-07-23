import time
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

from GAT.parser import parse_args
from GAT.models import GAT
from GAT.load_data import Data

# Settings
args = parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset, adj_type=args.adj_type, test_size=args.test_size, seed=args.seed)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature
label = data.label
# adj = data.adj
idx_train = data.idx_train
idx_val = data.idx_val
idx_test = data.idx_test
edge_list = data.edge_list


"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GAT(n_feature=feature.shape[1], n_hidden=args.hidden, n_class=data.n_class, dropout=args.dropout, n_head=args.n_head)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(feature, edge_list)
    loss_train = F.nll_loss(input=output[idx_train], target=label[idx_train])
    acc_train = accuracy_score(y_pred=output[idx_train].max(1)[1], y_true=label[idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(feature, edge_list)
    loss_val = F.nll_loss(input=output[idx_val], target=label[idx_val])
    acc_val = accuracy_score(y_pred=output[idx_val].max(1)[1], y_true=label[idx_val])

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train.item() ,loss_val.item(), acc_train, acc_val))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(feature, edge_list)
loss_test = F.nll_loss(input=output[idx_test], target=label[idx_test])
acc_test = accuracy_score(y_pred=output[idx_test].max(1)[1], y_true=label[idx_test])
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test.item(), acc_test))
