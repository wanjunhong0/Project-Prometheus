# Project-Prometheus

## Classic GNN methods (Pytorch Implementation)

### Package Requirements
```
torch
torch_geometric
torchmetrics
scipy
numpy
```
### Universal Parameters
- Data Paramters
```
--data_path     str     '../data/'   Input data path
--dataset       str     'Cora'       Choose a dataset from {Cora, CiteSeer, PubMed}
--split         str     'full'       The type of dataset split {public, full, random}
```
- Training Parameters
```
--seed          int      123         Random seed 
--epoch         int      100         Number of epochs to train
--lr            float    0.01        Initial learning rate
--weight_decay  float    5e-4        Weight decay (L2 norm on parameters)
--layer/--k     int/int  2/10        Number of layers/k-hop propagations
--hidden        int      64          Number of hidden units
--dropout       float    0.5         Dropout rate
``` 
NOTE: `--layer` indicates numbers of (Graph) Neural Networks layers which include parameters; while `--k` indicates k-hop propagations which do not have parameters


### Model Parameters
- ChebNet [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)

- GCN [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

- GraphSAGE [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
```
--sample        int      10          Number of neighbors to sample (0 means no sampling)
--batch_size    int      128         Number of nodes per batch
---aggregator   str      mean        Choose a aggregator type from {mean, gcn, meanpooling}
```

- GAT [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
```
--batch_size    int      128         Number of nodes per batch
--head          int      8           Number of head attentions
```

- FastGCN [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247)
```
--batch_size    int      128         Number of nodes per batch
--sample        int      256         Number of neighbors to sample per layer
```

- SIGN [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198)

- SGC [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)

- JKnet [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536)
```
--mode          str      cat         Mode for layer fusion {cat, max}
```

- DropEdge [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://arxiv.org/abs/1907.10903)
```
--rate          float    0.8         The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix
```

- PairNorm [PairNorm: Tackling Oversmoothing in GNNs](https://arxiv.org/abs/1909.12223)
```
--scale         float    1.0         Row-normalization scale
--mode          str      PN          Mode for PairNorm {None, PN, PN-SI, PN-SCS}
```

- DAGNN [Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296)

- GCNII [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133)
```
--alpha         float    0.1         Alpha hyperparameters
--theta         float    0.5         Theta hyperparameters
```

- APPNP [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)
```
--alpha         float    0.1         Alpha hyperparameters
```

- GPRGNN [Adaptive Universal Generalized PageRank Graph Neural Network](https://arxiv.org/abs/2006.07988)
```
--alpha         float    0.1         Alpha hyperparameters
```

- FAGCN [Beyond Low-frequency Information in Graph Convolutional Networks](https://arxiv.org/abs/2101.00797)
```
--eps           float    0.3         Eps hyperparameters
```

- BertNet [BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation](https://arxiv.org/abs/2106.10994)

### Examples
```
cd GCN
python main.py
```