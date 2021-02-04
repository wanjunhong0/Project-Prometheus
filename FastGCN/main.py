import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from FastGCN.models import FastGCN
from FastGCN.parser import parse_args
from FastGCN.load_data import Data, Dataset
from FastGCN.sampler import Sampler


# Settings
args = parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
device = torch.device(args.device)

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset, adj_type=args.adj_type, val_test=args.val_test)
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
model = FastGCN(n_feature=feature.shape[1], n_hidden=args.hidden, n_class=data.n_class, dropout=args.dropout).to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    loss_train, acc_train = 0, 0
    for idx_batch in train_loader:
        optimizer.zero_grad()
        feature_, adj_layer1, adj_layer2 = sampler.sampling(idx_batch)
        print(time.time()-t)
        output = model(feature_.to(device), adj_layer1.to(device), adj_layer2.to(device)).cpu()
        loss = F.nll_loss(input=output, target=label[idx_batch])  # mean loss to backward
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(idx_batch)
        acc_train += accuracy_score(y_pred=output.max(1)[1], y_true=label[idx_batch], normalize=False)
    loss_train, acc_train = [x / len(train) for x in [loss_train, acc_train]]

    # Validation
    model.eval()
    output = model(feature.to(device), norm_adj.to(device), norm_adj.to(device)).cpu()
    loss_val = F.nll_loss(input=output[idx_val], target=label[idx_val])
    acc_val = accuracy_score(y_pred=output[idx_val].max(1)[1], y_true=label[idx_val])

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))
