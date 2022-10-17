from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from self_supervised_InIModel import InIModel
import dgl
import networkx as nx
from evaluation import node_classification_evaluation

class GM:
    # gradient matching class
    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        
        # GCond parameters
        
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        # TODO PGE to generate adj_syn
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)
        
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        
    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        
    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.
        if args.dataset in ['ogbn-arxiv']:
            return args.outer, args.inner
        if args.dataset in ['cora']:
            return 20, 15 # sgc
        if args.dataset in ['citeseer']:
            return 20, 15
        if args.dataset in ['physics']:
            return 20, 10
        else:
            return 20, 10
        
    def train(self, verbose=True):
        args = self.args
        data = self.data
        self.num_nodes = self.data.feat_train.shape[0]
        
        feat_syn, pge = self.feat_syn, self.pge
        
        features, adj = data.feat_full, data.adj_full
        
        idx_train = data.idx_train
        
        features, adj = utils.to_tensor(features, adj, device=self.device)
        
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
            
        adj = adj_norm
        
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()
        
        outer_loop, inner_loop = self.get_loops(args)
        
        loss_avg = 0
        
        self.model = InIModel(data=self.data, args=self.args)
        self.model.to(self.device)
        self.final_acc_list = []
        self.estp_acc_list = []
        
        for it in range(args.epochs+1):
            model = self.model
            model_parameters = list(model.parameters())
            
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()
            
            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn
                
                loss = torch.tensor(0.0).to(self.device)
                
                # TODO graph to device and feature to device
                graph_orig = dgl.from_scipy(data.adj_full, device=self.device)
                train_mask = torch.full((data.nnodes,),False).index_fill_(0, torch.as_tensor(data.idx_train), True).to(self.device)
                val_mask = torch.full((data.nnodes,),False).index_fill_(0, torch.as_tensor(data.idx_val), True).to(self.device)
                test_mask = torch.full((data.nnodes,),False).index_fill_(0, torch.as_tensor(data.idx_test), True).to(self.device)
                graph_orig.ndata['train_mask'], graph_orig.ndata['test_mask'], graph_orig.ndata['val_mask'] = train_mask, test_mask, val_mask
                graph_orig.ndata['label'] = torch.as_tensor(data.labels_full).to(self.device)
                # print(graph.device)
                # graph = torch.tensor(data.adj_full).to(self.device)
                x_orig = torch.tensor(data.feat_full).to(self.device)
                
                loss_real, loss_item = self.model(graph_orig, x_orig)
                
                gw_real = torch.autograd.grad(loss_real, model_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                
                # TODO graph_syn and feature_syn to device                
                graph_syn = dgl.from_networkx(nx.from_numpy_array(adj_syn_norm.detach().cpu().numpy()), device=self.device)
                x_syn = feat_syn.to(self.device)
                
                loss_syn, loss_syn_item = self.model(graph_syn, x_syn)
                
                gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                
                # TODO coeff relaed (different labels bias)
                
                loss += match_loss(gw_syn, gw_real, args, device=self.device)
                
                #TODO alpha related regularize (label syn)
                
                loss += loss
                
                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                
                if it % 5 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()
                    
                if args.debug and ol % 5 == 0:
                    print('Gradient matching loss:', loss)
                    
                if ol == outer_loop - 1:
                    break
                
                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    #TODO feature and adj to device and fed into model
                    graph = dgl.from_networkx(nx.from_numpy_array(adj_syn_inner_norm.detach().cpu().numpy()), device=self.device)
                    x = feat_syn_inner_norm.to(self.device)
                    loss_syn_inner, loss_syn_inner_item = self.model(graph, x)
                    loss_syn_inner.backward()
                    optimizer_model.step()
            
            loss_avg /= outer_loop
            if it % 5 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))
                
            # TODO evaluation
            eval_epochs = [1, 20]
            
            if it in eval_epochs:
                print('******************** eval*******************')
                final_acc, estp_acc = node_classification_evaluation(self.model, graph_orig, x_orig, self.args.num_classes, self.args.lr_adj_f,self.args.weight_decay_f, self.args.max_epoch_f, self.device, self.args.linear_prob)
                self.final_acc_list.append(final_acc)
                self.estp_acc_list.append(estp_acc)
                
                
                