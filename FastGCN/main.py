import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from FastGCN.models import FastGCN
from FastGCN.parser import parse_args
from FastGCN.load_data import Data


# Settings
args = parse_args()
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
data = Data(path=args.data_path + args.dataset, adj_type=args.adj_type, val_test=args.val_test, seed=args.seed)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature.to(device)
norm_adj = data.norm_adj.to(device)
label = data.label
idx_train = data.idx_train
idx_val = data.idx_val
idx_test = data.idx_test

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = FastGCN(n_feature=feature.shape[1], n_hidden=args.hidden, n_class=data.n_class, dropout=args.dropout)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(feature, norm_adj).cpu()
    loss_train = F.nll_loss(input=output[idx_train], target=label[idx_train])
    acc_train = accuracy_score(y_pred=output[idx_train].max(1)[1], y_true=label[idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(feature, norm_adj).cpu()
    loss_val = F.nll_loss(input=output[idx_val], target=label[idx_val])
    acc_val = accuracy_score(y_pred=output[idx_val].max(1)[1], y_true=label[idx_val])

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train.item(), loss_val.item(), acc_train, acc_val))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(feature, norm_adj).cpu()
loss_test = F.nll_loss(input=output[idx_test], target=label[idx_test])
acc_test = accuracy_score(y_pred=output[idx_test].max(1)[1], y_true=label[idx_test])
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test.item(), acc_test))
