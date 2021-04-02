import time
import torch


a = torch.eye(10000).to_sparse().coalesce()
t = time.time()
c = a.index_select(0, torch.arange(1000))
print(time.time()-t)
b = []
for i in range(1000):
    b.append(a[i])
torch.stack(b)
print(b== c)
print(time.time()-t)
