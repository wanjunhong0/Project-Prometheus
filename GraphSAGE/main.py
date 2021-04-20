import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from models import SupervisedGraphSAGE
from load_data import Data, Dataset
from sampler import Sampler

"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser(description="Run GraphSAGE.")
parser.add_argument('--data_path', nargs='?', default='../data/', help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='Cora', help='Choose a dataset')
parser.add_argument('--split', nargs='?', default='full', help='The type of dataset split {public, full, random}')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--sample', type=int, default=10, help='Number of neighbors to sample (0 means no sampling).')
parser.add_argument('--batch_size', type=int, default=128, help='Number of sample per batch.')
parser.add_argument('--aggregator', nargs='?', default='mean', help='Choose a aggregator type from {mean, gcn, meanpooling}')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
# training on the first GPU if not on CPU
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
train_loader = DataLoader(dataset=train, batch_size=args.batch_size)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size)
test_loader = DataLoader(dataset=test, batch_size=args.batch_size)
sampler = Sampler(data.adj, args.aggregator)

"""
===========================================================================
Training
===========================================================================
"""
model = SupervisedGraphSAGE(n_feature=data.n_feature, n_hidden=args.hidden, n_class=data.n_class,
                            agg_type=args.aggregator, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
metric = torchmetrics.Accuracy().to(device)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    loss_train = 0
    metric.reset()
    for idx_batch in train_loader:
        optimizer.zero_grad()
        label_batch = label[idx_batch]
        adj1_batch, adj2_batch = sampler.sampling(idx_batch, args.sample)
        output = model(feature, adj1_batch.to(device), adj2_batch.to(device))[idx_batch]
        loss_batch = F.nll_loss(output, label_batch)  # mean loss to backward
        loss_batch.backward()
        optimizer.step()
        loss_train += loss_batch * len(idx_batch)
        acc_batch = metric(output.max(1)[1], label_batch)
    loss_train = loss_train / len(data.idx_train)
    acc_train = metric.compute()

    # Validation
    model.eval()
    loss_val = 0
    metric.reset()
    for idx_batch in val_loader:
        label_batch = label[idx_batch]
        adj1_batch, adj2_batch = sampler.sampling(idx_batch, 0)
        output = model(feature, adj1_batch.to(device), adj2_batch.to(device))[idx_batch]
        loss_val += F.nll_loss(output, label_batch, reduction='sum')
        acc_batch = metric(output.max(1)[1], label_batch)
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
for idx_batch in test_loader:
    label_batch = label[idx_batch]
    adj1_batch, adj2_batch = sampler.sampling(idx_batch, 0)
    output = model(feature, adj1_batch.to(device), adj2_batch.to(device))[idx_batch]
    loss_test += F.nll_loss(output, label_batch, reduction='sum')
    acc_batch = metric(output.max(1)[1], label_batch)
loss_test = loss_test / len(data.idx_test)
acc_test = metric.compute()

print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test, acc_test))
