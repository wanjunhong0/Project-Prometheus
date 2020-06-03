import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from GCN.utils import load_data, accuracy
from GCN.models import GCN
from GCN.parser import parse_args


# Settings
args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

"""
===========================================================================
Loading data
===========================================================================
"""

adj, features, labels, idx_train, idx_val, idx_test = load_data()


"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GCN(n_features=features.shape[1],
            n_hiddens=args.hidden,
            n_class=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epochs+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))


# # Train model
# t_total = time.time()
# # for epoch in range(args.epochs):
# #     train(epoch)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Testing
# test()
