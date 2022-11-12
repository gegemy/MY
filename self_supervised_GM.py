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
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from self_supervised_InIModel import InIModel
import dgl
import networkx as nx
from evaluation import node_classification_evaluation, syn_train

class GM:
    # gradient matching class
    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        
        self.nnodes = data.nnodes
        
        # GCond parameters
        
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        # TODO PGE to generate adj_syn
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)
        
        self.cluster_syn = torch.LongTensor(torch.arange(self.nnodes_syn)).to(device)
        
        from collections import Counter;
        counter = Counter(data.cluster_full)
        self.num_cluster_dict = dict(counter)
        
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

    def preprocess(self, graph):
        graph = graph.cpu()
        feat = graph.ndata["feat"]
        graph = dgl.to_bidirected(graph)
        graph.ndata["feat"] = feat

        graph = graph.remove_self_loop().add_self_loop()
        graph.create_formats_()
        return graph
    
    def transferTodgl(self, data):
        graph_orig = dgl.from_scipy(data.adj_full).cpu()
        graph_orig.ndata['feat'] = torch.tensor(data.feat_full).cpu()
        graph_orig = self.preprocess(graph_orig)
        graph_orig = graph_orig.to(self.device)
        x_orig = graph_orig.ndata['feat'].to(self.device)
        train_mask = torch.full((data.nnodes,),False).index_fill_(0, torch.as_tensor(data.idx_train), True).to(self.device)
        val_mask = torch.full((data.nnodes,),False).index_fill_(0, torch.as_tensor(data.idx_val), True).to(self.device)
        test_mask = torch.full((data.nnodes,),False).index_fill_(0, torch.as_tensor(data.idx_test), True).to(self.device)
        graph_orig.ndata['train_mask'], graph_orig.ndata['test_mask'], graph_orig.ndata['val_mask'] = train_mask, test_mask, val_mask
        graph_orig.ndata['label'] = torch.as_tensor(data.labels_full).to(self.device)
        graph_orig.ndata['cluster'] = torch.as_tensor(data.cluster_full).to(self.device)
        graph_orig.cluster_num = self.nnodes_syn
        return graph_orig, x_orig
    
    def get_sub_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.cluster_syn.cpu().numpy())

        for c in range(self.nnodes_syn):
            tmp = data.retrieve_cluster(c, self.nnodes_syn, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]
        return features
        
    def train(self, verbose=True):
        args = self.args
        data = self.data
        

        feat_syn, pge = self.feat_syn, self.pge
        feat_sub  = torch.FloatTensor(self.get_sub_feat(data.feat_full))
        self.feat_syn.data.copy_(feat_sub)
        
        # TODO graph to device and feature to device
        graph_orig, x_orig = self.transferTodgl(data)
        
        outer_loop, inner_loop = self.get_loops(args)
        
        loss_avg = 0
        
        # adj = data.adj_full
        # features = data.feat_full
        # features, adj = utils.to_tensor(features, adj, device=self.device)
            
        # if utils.is_sparse_tensor(adj):
        #     adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        # else:
        #     adj_norm = utils.normalize_adj_tensor(adj)

        # adj = adj_norm
        # adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
        #         value=adj._values(), sparse_sizes=adj.size()).t()     
        self.final_acc_list = []
        self.estp_acc_list = []
        
        # using the same model for training
        # model = InIModel(data=self.data, args=self.args)
        # model.to(self.device)
        # model_parameters = list(model.parameters())
        
        for it in range(args.epochs+1):
            # print('model parameters:')
            # for tmp in model.parameters():
            #     print(tmp.shape)
            # exit()
            
            # init model for each epoch
            # model = InIModel(data=self.data, args=self.args)
            # model.to(self.device)
            # model_parameters = list(model.parameters())
            
            if it % 200 == 0:
                # period init model for training
                model = InIModel(data=self.data, args=self.args)
                model.to(self.device)
                model_parameters = list(model.parameters())
            
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()
            
            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn
                
                loss = torch.tensor(0.0).to(self.device)
                ls = 0
                lreal = 0
                
                # Calculate loss with whole graph                
                loss_real_full = model(graph_orig, x_orig, gtype='orig')
                
                # Calculate loss with syn --- mask a batch nodes
                graph_syn = dgl.from_networkx(nx.from_numpy_array(adj_syn_norm.detach().cpu().numpy()), device=self.device)
                graph_syn.ndata['cluster'] = self.cluster_syn.to(self.device)
                x_syn = feat_syn.to(self.device)
                    
                loss_syn_full = model(graph_syn, x_syn, gtype='syn')
                print(type(loss_syn_full))    
                
                for c in loss_syn_full.keys():
                    # print(c)
                    loss_real = loss_real_full[c]
                    lreal += loss_real
                    gw_real = torch.autograd.grad(loss_real, model_parameters, retain_graph=True)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    
                    loss_syn = loss_syn_full[c]
                    ls += loss_syn
                    
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True, retain_graph=True)
                    coeff = self.num_cluster_dict[c.item()] / max(self.num_cluster_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)        
                
                # for c in range(self.nnodes_syn):
                #     loss_real = loss_real_full[c]
                #     lreal += loss_real
                #     gw_real = torch.autograd.grad(loss_real, model_parameters, retain_graph=True)
                #     gw_real = list((_.detach().clone() for _ in gw_real))
                    
                #     graph_syn = dgl.from_networkx(nx.from_numpy_array(adj_syn_norm.detach().cpu().numpy()), device=self.device)
                #     graph_syn.ndata['cluster'] = self.cluster_syn.to(self.device)
                #     x_syn = feat_syn.to(self.device)
                    
                #     loss_syn = model(graph_syn, x_syn, c, gtype='syn', mask_rate=1.0)
                #     ls += loss_syn
                    
                #     gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                #     coeff = self.num_cluster_dict[c] / max(self.num_cluster_dict.values())
                #     loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)
                    
                print('Epoch {}, outer_loop {}, loss real {}'.format(it, ol, lreal))
                print('Epoch {}, outer_loop {}, loss syn {}'.format(it, ol, ls))
                print('Epoch {}, outer_loop {}, loss {}'.format(it, ol, loss))

                # # Calculate loss with whole graph                
                # loss_real, loss_item = model(graph_orig, x_orig)
                # print('Epoch {}, outer_loop {}, loss real {}'.format(it, ol, loss_real))
                
                # gw_real = torch.autograd.grad(loss_real, model_parameters)
                # gw_real = list((_.detach().clone() for _ in gw_real))
                
                # model.zero_grad()
                
                # # TODO graph_syn and feature_syn to device                
                # graph_syn = dgl.from_networkx(nx.from_numpy_array(adj_syn_norm.detach().cpu().numpy()), device=self.device)
                # x_syn = feat_syn.to(self.device)
                
                # loss_syn, loss_syn_item = model(graph_syn, x_syn)
                # print('Epoch {}, outer_loop {}, loss syn {}'.format(it, ol, loss_syn))
                
                # gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                
                # # TODO coeff related (different labels bias)
                
                # loss += match_loss(gw_syn, gw_real, args, device=self.device)
                # print('Epoch {}, outer_loop {}, loss {}'.format(it, ol, loss))
                loss_avg += loss.item()
                
                #TODO alpha related regularize (label syn)
                    
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
                    graph_syn_norm = dgl.from_networkx(nx.from_numpy_array(adj_syn_inner_norm.detach().cpu().numpy()), device=self.device)
                    x_syn_norm = feat_syn_inner_norm.to(self.device)
                    loss_syn_inner = model(graph_syn_norm, x_syn_norm)
                    loss_syn_inner.backward()
                    optimizer_model.step()
            
            loss_avg /= (self.nnodes_syn*outer_loop)
            if it % 5 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))
                
            
            if it % 100 == 0:
                print('******************** EVAL Epoch *******************')
                #TODO
                #init new GraphMAE model
                evalModel = InIModel(data=self.data, args=self.args)
                evalModel.to(self.device)
                evalmodel_parameters = list(evalModel.parameters())
                evaloptimizer_model = torch.optim.Adam(evalmodel_parameters, lr=args.lr_model)
                
                #using syn to train model, using orig to evaluate model
                final_acc, estp_acc = syn_train(evalModel, graph_orig, graph_syn, x_orig, x_syn, evaloptimizer_model, self.device, self.args.num_classes, self.args.lr_adj_f, self.args.weight_decay_f, self.args.max_epoch_f, self.args.linear_prob, self.args.max_epoch_s)
                
                self.final_acc_list.append(final_acc)
                self.estp_acc_list.append(estp_acc)
            
            model.zero_grad()
                
                
                