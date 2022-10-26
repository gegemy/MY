from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from self_supervised_GM import GM
from utils_graphsaint import DataGraphSAINT


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--mask_rate', type=float, default=0.5)
parser.add_argument('--encoder', type=str, default='gat')
parser.add_argument('--decoder', type=str, default='gat')
parser.add_argument('--in_drop', type=float, default=0.2, help="input feature dropout")
parser.add_argument('--attn_drop', type=float, default=0.1, help="attention dropout")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--nlayers', type=int, default=3) #num_layer
parser.add_argument('--hidden', type=int, default=256) # num_hidden
parser.add_argument('--heads', type=int, default=4, help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
parser.add_argument('--max_epoch', type=int, default=1000, help="number of training epochs")
parser.add_argument('--max_epoch_f', type=int, default=600, help="for evaluation")
parser.add_argument('--max_epoch_s', type=int, default=1000, help="for syn train")

parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)

parser.add_argument('--lr_adj_f', type=float, default=0.01)
parser.add_argument('--lr_feat_f', type=float, default=0.01)
parser.add_argument('--lr_model_f', type=float, default=0.01)

parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--weight_decay_f', type=float, default=0.0)

parser.add_argument('--activation', type=str, default='prelu')
parser.add_argument('--optimizer', type=str, default='adam')

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--drop_edge_rate', type=float, default=0.0)
parser.add_argument('--loss_fn', type=str, default='sce')
parser.add_argument('--reduction_rate', type=float, default=0.001)
parser.add_argument('--replace_rate', type=float, default=0.05)
parser.add_argument('--alpha_l', type=int, default=3, help="`pow`inddex for `sce` loss")
parser.add_argument('--seed', type=int, default=15, help='Random seed.')


parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--scheduler', type=bool, default=True)
parser.add_argument('--norm', type=str, default=None)

parser.add_argument('--keep_ratio', type=float, default=1.0)
# parser.add_argument('--nsamples', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=1)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=0)
parser.add_argument("--concat_hidden", default=False)
parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
parser.add_argument("--residual", default=False,
                        help="use residual connection")
parser.add_argument("--linear_prob", default=False)
args = parser.parse_args()

torch.cuda.set_device(args.device)

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

args.num_features = data.feat_full.shape[1]
args.num_classes = data.nclass
print('num classes is: {}'.format(args.num_classes))

device = args.device if args.device >= 0 else "cpu"
model = GM(data=data, args=args, device=device)

model.train()
