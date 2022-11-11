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
import scipy.sparse as sp
from torch_sparse import SparseTensor
from utils import drop_edge
from models.gat_mae import GAT
from models.loss_func import sce_loss
from utils import create_norm
from functools import partial
import copy

def setup_module(model_type, args) -> nn.Module:
    if args.encoder in ("gat"):
        enc_num_hidden = args.hidden // args.heads
        enc_nhead = args.heads
    else:
        enc_num_hidden = args.hidden
        enc_nhead = 1
        
    dec_in_dim = args.hidden
    dec_num_hidden = args.hidden // args.num_out_heads if args.decoder in ("gat") else args.hidden
    
    in_dim = args.num_features
    num_hidden = args.hidden
    num_layers = args.nlayers
    nhead = args.heads
    nhead_out = args.num_out_heads
    activation = args.activation
    feat_drop = args.in_drop
    attn_drop = args.attn_drop
    mask_rate = args.mask_rate
    norm = create_norm(args.norm)
    loss_fn = args.loss_fn
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    alpha_l = args.alpha_l
    concate_hidden = args.concat_hidden
    negative_slope=args.negative_slope
    residual=args.residual
    concat_out = True
    
    # print(model_type)
    
    if model_type == 'encoding':
        print('in_dim:{}, num_hidden:{},out_dim:{},num_layers:{},nhead:{},nhead_out:{}, concat_out:{}'.format(in_dim, enc_num_hidden, enc_num_hidden, num_layers, enc_nhead, enc_nhead, concat_out))
        model = GAT(in_dim=in_dim, num_hidden=enc_num_hidden,out_dim=enc_num_hidden,
                    num_layers=num_layers,nhead=enc_nhead, nhead_out=enc_nhead, concat_out=concat_out,
                    activation=activation,feat_drop=feat_drop,attn_drop=attn_drop,
                    negative_slope=negative_slope, residual=residual,norm=norm, encoding=(model_type == "encoding"),
                    )
    elif model_type == 'decoding':
        print('in_dim:{}, num_hidden:{},out_dim:{},num_layers:{},nhead:{},nhead_out:{}, concat_out:{}'.format(dec_in_dim, dec_num_hidden, in_dim, 1, nhead, nhead_out, concat_out))
        model = GAT(in_dim=dec_in_dim, num_hidden=dec_num_hidden, out_dim=in_dim,
                    num_layers=1, nhead=nhead, nhead_out=nhead_out, activation=activation,
                    feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                    residual=residual, norm=norm, concat_out=True, encoding=(model_type == "encoding"),)
    else:
        raise NotImplementedError
    return model

class InIModel(nn.Module):
    def __init__(self, data, args, device='cuda', **kwargs):
        super(InIModel, self).__init__()
        self.data = data
        self.args = args
        self.device = args.device
        
        # GraphMAE parameters
        
        self._mask_rate = args.mask_rate
        
        self._encoder_type = args.encoder
        self._decoder_type = args.decoder
        
        self._drop_edge_rate = args.drop_edge_rate
        self._output_hidden_size = args.hidden
        self._concat_hidden = args.concat_hidden
        
        self._replace_rate = args.replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        
        assert args.hidden % args.heads == 0
        assert args.hidden % args.num_out_heads == 0
        # if self._encoder_type in ("gat"):
        #     self.enc_num_hidden = args.hidden // args.heads
        #     self.enc_nhead = args.heads
        # else:
        #     self.enc_num_hidden = args.hidden
        #     self.enc_nhead = 1
            

        # self.dec_num_hidden = args.hidden // args.num_out_heads if self._decoder_type in ("gat") else args.hidden
        
        # build encoder
        # TODO 
        self.encoder =  setup_module(model_type='encoding', args=args)
        
        # TODO for attribute prediction to predict 
        self.decoder = setup_module(model_type='decoding', args=args)
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, args.num_features)).to(args.device)
        self.dec_in_dim = args.hidden
        if args.concat_hidden:
            self.encoder_to_decoder = nn.Linear(self.dec_in_dim * args.nlayers, self.dec_in_dim, bias=False)
            
        else:
            self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
            
        # setup loss fuction
        self.criterion = self.setup_loss_fn(args.loss_fn, args.alpha_l)
    
    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
    
    #TODO loss function (sce part)
    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == 'mse':
            croteropm = nn.MESLoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
        
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device = x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
            
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        
        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def encoding_syn_mask_noise(self, g, x, c, mask_rate):
        num_nodes = torch.numel(g.ndata['cluster'][g.ndata['cluster']==c])
        nodes = (g.ndata['cluster']==c).nonzero(as_tuple=True)[0]        
        perm = torch.randperm(num_nodes, device=x.device)
        perm = nodes.view(-1)[perm].view(nodes.size())
        num_mask_nodes = int(mask_rate * num_nodes)        
        mask_nodes = perm[: num_mask_nodes]
        
        all_nodes = np.arange(g.num_nodes())
        mask = mask_nodes.cpu().numpy()
        keep_nodes = torch.tensor(np.delete(all_nodes, mask), device=x.device)
        
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0
            
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        
        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def encoding_cluster_mask_noise(self, g, x, mask_rate):
        print('mask rate is:{}'.format(mask_rate))
        mask_cluster_dic = dict()
        for c in range(g.cluster_num):
            num_nodes = torch.numel(g.ndata['cluster'][g.ndata['cluster']==c])
            nodes = (g.ndata['cluster']==c).nonzero(as_tuple=True)[0]        
            perm = torch.randperm(num_nodes, device=x.device)
            perm = nodes.view(-1)[perm].view(nodes.size())
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            mask_cluster_dic[c] = mask_nodes
            if c == 0:
                all_mask_nodes = copy.deepcopy(mask_nodes)  
            else:
                all_mask_nodes = torch.cat((all_mask_nodes, mask_nodes))
        
        all_nodes = np.arange(g.num_nodes())
        mask = all_mask_nodes.cpu().numpy()
        keep_nodes = torch.tensor(np.delete(all_nodes, mask), device=x.device)
        
        out_x = x.clone()
        token_nodes = all_mask_nodes
        out_x[all_mask_nodes] = 0.0
            
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        
        return use_g, out_x, (all_mask_nodes, keep_nodes), mask_cluster_dic
    
    def mask_attr_prediction(self, g, x, c=None, gtype=None, mask_rate=None):
        mask_cluster_dic = None
        if gtype == None:
            pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        elif gtype == 'orig':
        # mask half nodes in the specific class(for orig), and mask all nodes in the specific class for syn
            pre_use_g, use_x, (mask_nodes, keep_nodes), mask_cluster_dic = self.encoding_cluster_mask_noise(g, x, self._mask_rate)
        elif gtype == 'syn':
            pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_syn_mask_noise(g, x, c, mask_rate=1)
        else:
            print('ERROR for attr mask')
            exit()
        if self._drop_edge_rate > 0:
            use_g, mask_edgess = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
            
        
        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
            
        # ----- attribute reconstruction
        rep = self.encoder_to_decoder(enc_rep)
        
        if self._decoder_type != 'mlp':
            rep[mask_nodes] = 0
            
        if self._decoder_type == 'mlp':
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)
        
        if mask_cluster_dic != None:
            loss = []
            for c in range(g.cluster_num):
                x_init = x[mask_cluster_dic[c]]
                x_rec = recon[mask_cluster_dic[c]]
                tmp_loss = self.criterion(x_rec, x_init)
                loss.append(tmp_loss)
        else:
            x_init = x[mask_nodes]
            x_rec = recon[mask_nodes]
            loss = self.criterion(x_rec, x_init)
        return loss
    
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep
    
    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
            
    
    # TODO forward
    def forward(self, g, x, c=None, gtype=None, mask_rate=0.3):
        loss = self.mask_attr_prediction(g, x, c, gtype, mask_rate)
        return loss
        

            
        
        
        
        
    
    
    