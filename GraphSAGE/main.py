import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from GraphSAGE.models import SupervisedGraphSAGE
from GraphSAGE.parser import parse_args
from GraphSAGE.load_data import Data, Dataset, Collate


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
data = Data(path=args.data_path + args.dataset, test_size=args.test_size, seed=args.seed)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = data.feature.to(device)
label = data.label
train = Dataset(data.neighbor_list, data.idx_train)
val = Dataset(data.neighbor_list, data.idx_val)
test = Dataset(data.neighbor_list, data.idx_test)
collate = Collate(data.neighbor_list, args.sample, args.aggregator)
train_loader = DataLoader(dataset=train, batch_size=args.batch_size, collate_fn=collate)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size, collate_fn=collate)
test_loader = DataLoader(dataset=test, batch_size=args.batch_size, collate_fn=collate)

"""
===========================================================================
Training
===========================================================================
"""
model = SupervisedGraphSAGE(n_feature=feature.shape[1], n_hidden=args.hidden, n_class=data.n_class, 
                            agg_type=args.aggregator, dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.to(device)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    loss_train, acc_train = 0, 0
    for adj_batch, idx_batch in train_loader:
        optimizer.zero_grad()
        output = model(feature, adj_batch.to(device)).cpu()
        loss = F.nll_loss(input=output[idx_batch], target=label[idx_batch])  # mean loss to backward
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(idx_batch)
        acc_train += accuracy_score(y_pred=output[idx_batch].max(1)[1], y_true=label[idx_batch], normalize=False)
    loss_train, acc_train = [x / len(train) for x in [loss_train, acc_train]]

    # Validation
    model.eval()
    loss_val, acc_val = 0, 0
    for adj_batch, idx_batch in val_loader:
        output = model(feature, adj_batch.to(device)).cpu()
        loss_val += F.nll_loss(input=output[idx_batch], target=label[idx_batch], reduction='sum')
        acc_val += accuracy_score(y_pred=output[idx_batch].max(1)[1], y_true=label[idx_batch], normalize=False)
    loss_val, acc_val = [x / len(val) for x in [loss_val, acc_val]]

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))

"""
===========================================================================
Testing
===========================================================================
"""
loss_test, acc_test = 0, 0
for adj_batch, idx_batch in test_loader:
    output = model(feature, adj_batch.to(device)).cpu()
    loss_test += F.nll_loss(input=output[idx_batch], target=label[idx_batch], reduction='sum')
    acc_test += accuracy_score(y_pred=output[idx_batch].max(1)[1], y_true=label[idx_batch], normalize=False)
loss_test, acc_test = [x / len(test) for x in [loss_test, acc_test]]

print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test, acc_test))
