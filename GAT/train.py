import time
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

from GAT.parser import parse_args
from GAT.utils import load_data, accuracy
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
norm_adj = data.norm_adj
idx_train = data.idx_train
idx_val = data.idx_val
idx_test = data.idx_test


# nomr_adj, feature, label, idx_train, idx_val, idx_test = load_data()

model = GAT(nfeat=feature.shape[1], nhid=args.hidden, nclass=data.n_class, dropout=args.dropout, nheads=args.n_head)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(feature, norm_adj)
    loss_train = F.nll_loss(output[idx_train], label[idx_train])
    acc_train = accuracy(output[idx_train], label[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], label[idx_val])
    acc_val = accuracy(output[idx_val], label[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(feature, norm_adj)
    loss_test = F.nll_loss(output[idx_test], label[idx_test])
    acc_test = accuracy(output[idx_test], label[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []

for epoch in range(args.epoch):
    loss_values.append(train(epoch))



print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



# Testing
compute_test()