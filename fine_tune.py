from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcond_agent_transduct import GCond
from utils_graphsaint import DataGraphSAINT
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import torch.nn as nn
from models.gcn import GCN
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils

def generate_labels_syn(data):
    from collections import Counter
    counter = Counter(data.labels_train)
    num_class_dict = {}
    n = len(data.labels_train)

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    labels_syn = []
    syn_class_indices = {}
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * args.reduction_rate) - sum_
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

    num_class_dict = num_class_dict
    return labels_syn

def reset_parameters():
    feat_syn.data.copy_(torch.randn(feat_syn.size()))

def test_with_val(data, feat_syn, verbose=True):
    res = []

    data, device = data, 'cuda'
    feat_syn = feat_syn.detach()

    # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
    model = GCN(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=0.5,
                weight_decay=5e-4, nlayers=2,
                nclass=data.nclass, device=device).to(device)

    if args.dataset in ['ogbn-arxiv']:
        model = GCN(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=0.5,
                    weight_decay=0e-4, nlayers=2, with_bn=False,
                    nclass=data.nclass, device=device).to(device)

    adj_syn = pge.inference(feat_syn)

    adj_syn = torch.load(f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
    feat_syn = torch.load(f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

    if args.lr_adj == 0:
        n = len(labels_syn)
        adj_syn = torch.zeros((n, n))

    model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                    train_iters=600, normalize=True, verbose=False, report=False)

    model.eval()

    # ************ fine tune model **********
    start_time = time.time()
    # labels = data.labels_full
    model.fit_with_val(data.feat_train, data.adj_train, data.labels_train, data,
                    train_iters=args.train_iters, normalize=True, verbose=False, initialize=False, report=True)

    model.eval()
    
    labels_test = torch.LongTensor(data.labels_test).cuda()

    labels_train = torch.LongTensor(data.labels_train).cuda()
    print('shape of data.feat_train is:{}'.format(data.feat_train.shape))
    print('shape of data.adj_train is:{}'.format(data.adj_train.shape))
    output = model.predict(data.feat_train, data.adj_train)
    loss_train = F.nll_loss(output, labels_train)
    acc_train = utils.accuracy(output, labels_train)
    if verbose:
        print("Train set results:",
                "loss= {:.4f}".format(loss_train.item()),
                "accuracy= {:.4f}".format(acc_train.item()))
    res.append(acc_train.item())

    # Full graph
    print('shape of data.feat_full is:{}'.format(data.feat_full.shape))
    print('shape of data.adj_full is:{}'.format(data.adj_full.shape))
    output = model.predict(data.feat_full, data.adj_full)
    loss_test = F.nll_loss(output[data.idx_test], labels_test)
    acc_test = utils.accuracy(output[data.idx_test], labels_test)
    res.append(acc_test.item())
    if verbose:
        print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
    stop_time = time.time()
    print("Fine tune time:{} second".format(stop_time - start_time))
    


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
# parser.add_argument('--nsamples', type=int, default=20)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--train_iters', type=int, default=600)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    
print(type(data_full))
print(data_full.adj.shape)
print(data.labels_full.shape)

# labels = np.concatenate((data.labels_train, data.labels_test, data.labels_val), axis=0)
# print(labels.shape)
# exit()


n = int(data.feat_train.shape[0] * args.reduction_rate)
print('n is:{}'.format(n))
d = data.feat_train.shape[1]
print('d is:{}'.format(d))
feat_syn = nn.Parameter(torch.FloatTensor(n, d).to('cuda'))
pge = PGE(nfeat=d, nnodes=n, device='cuda',args=args).to('cuda')
labels_syn = torch.LongTensor(generate_labels_syn(data)).to('cuda')
reset_parameters()
optimizer_feat = torch.optim.Adam([feat_syn], lr=args.lr_feat)
optimizer_pge = torch.optim.Adam(pge.parameters(), lr=args.lr_adj)
print('adj_syn:', (n,n), 'feat_syn:', feat_syn.shape)
test_with_val(data, feat_syn)


# model = SGC1(nfeat=feat_syn.shape[1], nhid=args.hidden,
#             dropout=0.0, with_bn=False,
#             weight_decay=0e-4, nlayers=2,
#             nclass=data.nclass,
#             device='cuda').to('cuda')
# model.load_state_dict(torch.load(f'saved_model/model_{args.dataset}_{args.reduction_rate}_{args.seed}'))
# print(model)

