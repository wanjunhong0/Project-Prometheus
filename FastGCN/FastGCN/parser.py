import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run FastGCN.")
    parser.add_argument('--data_path', nargs='?', default='../data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='cora', help='Choose a dataset')
    parser.add_argument('--adj_type', nargs='?', default='symmetric', help='Graph laplacian type from {symmetric, unsymmetric}')

    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--val_test', nargs='+', type=float, default=[0.2, 0.4], help='The proportion of val and test set')
    parser.add_argument('--test_size', type=int, default=0.4, help='Test dataset size.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    return parser.parse_args()
