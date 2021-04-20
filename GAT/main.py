import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from models import GAT
from load_data import Data, Dataset, Collate


"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser(description="Run GAT.")
parser.add_argument('--data_path', nargs='?', default='../data/', help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='Cora', help='Choose a dataset')
parser.add_argument('--split', nargs='?', default='full', help='The type of dataset split {public, full, random}')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of sample per batch.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--head', type=int, default=8, help='Number of head attentions.')
args = parser.parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
# training on the first GPU if not available on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on device = {}'.format(device))

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path, dataset=args.dataset, split=args.split)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature.to(device)
label = data.label.to(device)
train = Dataset(data.idx_train)
val = Dataset(data.idx_val)
test = Dataset(data.idx_test)
collate = Collate(data.adj)
train_loader = DataLoader(dataset=train, batch_size=args.batch_size, collate_fn=collate)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size, collate_fn=collate)
test_loader = DataLoader(dataset=test, batch_size=args.batch_size, collate_fn=collate)

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GAT(n_feature=data.n_feature, n_hidden=args.hidden, n_class=data.n_class, dropout=args.dropout, n_head=args.head).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
metric = torchmetrics.Accuracy().to(device)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    loss_train = 0
    metric.reset()
    for edge_batch, idx_batch in train_loader:
        optimizer.zero_grad()
        output = model(feature, edge_batch.to(device))[idx_batch]
        loss_batch = F.nll_loss(output, label[idx_batch])  # mean loss to backward
        loss_batch.backward()
        optimizer.step()
        loss_train += loss_batch * len(idx_batch)
        acc_batch = metric(output.max(1)[1], label[idx_batch])
    loss_train = loss_train / len(data.idx_train)
    acc_train = metric.compute()

    # Validation
    model.eval()
    loss_val = 0
    metric.reset()
    for edge_batch, idx_batch in val_loader:
        output = model(feature, edge_batch.to(device))[idx_batch]
        loss_val += F.nll_loss(output, label[idx_batch], reduction='sum')
        acc_batch = metric(output.max(1)[1], label[idx_batch])
    loss_val = loss_val / len(data.idx_val)
    acc_val = metric.compute()

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))

"""
===========================================================================
Testing
===========================================================================
"""
loss_test = 0
metric.reset()
for edge_batch, idx_batch in test_loader:
    output = model(feature, edge_batch.to(device))[idx_batch]
    loss_test += F.nll_loss(output, label[idx_batch], reduction='sum')
    acc_batch = metric(output.max(1)[1], label[idx_batch])
loss_test = loss_test / len(data.idx_test)
acc_test = metric.compute()

print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test, acc_test))
