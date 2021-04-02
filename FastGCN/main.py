import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.metrics import accuracy_score

from models import FastGCN
from load_data import Data, Dataset
from sampler import Sampler


"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser(description="Run GCN.")
parser.add_argument('--data_path', nargs='?', default='../data/', help='Input data path')
parser.add_argument('--dataset', nargs='?', default='Cora', help='Choose a dataset from {Cora, CiteSeer, PubMed}')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, help='Number of sample per batch.')
parser.add_argument('--sample', type=int, default=128, help='Number of neighbors to sample per layer.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 norm on parameters)')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
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
data = Data(path=args.data_path, dataset=args.dataset)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature.to(device)
norm_adj = data.norm_adj.to(device)
label = data.label
idx_train = data.idx_train
idx_val = data.idx_val
idx_test = data.idx_test
train = Dataset(data.idx_train)
train_loader = DataLoader(dataset=train, batch_size=args.batch_size)
sampler = Sampler(data.feature, data.adj, args.sample)

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = FastGCN(n_feature=data.n_feature, n_hidden=args.hidden, n_class=data.n_class, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
metric = torchmetrics.Accuracy().to(device)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    feature_, adj_layer1, adj_layer2 = sampler.sampling(idx_train)
    output = model(feature_.to(device), adj_layer1.to(device), adj_layer2.to(device)).cpu()
    loss_train = F.nll_loss(output, label[idx_train])
    loss_train.backward()
    optimizer.step()
    acc_train = metric(output.max(1)[1], label[idx_train])

    # Validation
    model.eval()
    output = model(feature.to(device), norm_adj.to(device), norm_adj.to(device)).cpu()
    loss_val = F.nll_loss(output[idx_test], label[idx_test])
    acc_val = metric(output[idx_test].max(1)[1], label[idx_test])

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))
