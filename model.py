import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model import GCN, GAT
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from utils import *



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.mlp(x)


class MolGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gnn = GCN(in_feats=in_dim, hidden_feats=[hidden_dim, hidden_dim], allow_zero_in_degree=True)

    def forward(self, g, h):
        return self.gnn(g, h)



class MolMPNN(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.pre = nn.Linear(hidden_dim, hidden_dim)         
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, g, h):
        h = self.pre(h)                          
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h','msg'), fn.sum('msg','agg'))
        h_new = self.drop(self.mlp(g.ndata.pop('agg')))
        h = h + h_new if self.residual else h_new
        return self.norm(h)


class MolGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.h, self.nh, self.d, self.residual = hidden_dim, num_heads, hidden_dim // num_heads, residual
        self.q, self.k, self.v = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.norm_h, self.drop_h = nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        self.mlp_node = MLP(hidden_dim, hidden_dim, hidden_dim)
    def forward(self, g, h):
        Q, K, V = self.q(h).reshape(-1, self.nh, self.d), self.k(h).reshape(-1, self.nh, self.d), self.v(h).reshape(-1, self.nh, self.d)
        g.ndata.update({'Q': Q, 'K': K, 'V': V})
        g.apply_edges(lambda e: {'score': (e.src['K'] * e.dst['Q']).sum(-1) / (self.d ** 0.5), 'V': e.src['V']})
        g.edata['a'] = edge_softmax(g, g.edata['score'])
        g.edata['msg'] = g.edata['V'] * g.edata['a'].unsqueeze(-1)
        g.update_all(fn.copy_e('msg', 'msg'), fn.sum('msg', 'h_new'))
        h_new = self.mlp_node(g.ndata.pop('h_new').reshape(-1, self.nh * self.d))
        h = self.norm_h(h + self.drop_h(h_new)) if self.residual else self.norm_h(self.drop_h(h_new))
        return h

class Cov2MolGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.layer1 = MolGraphTransformer(hidden_dim, num_heads, dropout, residual)
        self.layer2 = MolGraphTransformer(hidden_dim, num_heads, dropout, residual)
    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        return h

class Cov2MolMPNN(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.layer1 = MolMPNN(hidden_dim, num_heads, dropout, residual)
        self.layer2 = MolMPNN(hidden_dim, num_heads, dropout, residual)
    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        return h

class ProtGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.h = hidden_dim
        self.nh = num_heads
        self.d = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.dropout_h = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)
        self.mlp_edge = MLP(hidden_dim * 3, hidden_dim, hidden_dim)
        self.mlp_node = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.residual = residual

    def forward(self, g, h, e):
        g.ndata['h'] = h
        g.edata['e'] = e
        g.apply_edges(lambda E: {'m': torch.cat([E.src['h'], E.data['e']], -1)})
        Q = self.q(h).reshape(-1, self.nh, self.d)
        K = self.k(g.edata['m']).reshape(-1, self.nh, self.d)
        V = self.v(g.edata['m']).reshape(-1, self.nh, self.d)
        g.ndata['Q'] = Q
        g.edata['K'] = K
        g.edata['V'] = V
        g.apply_edges(lambda E: {'s': (E.dst['Q'] * E.data['K']).sum(-1) / (self.d ** 0.5)})
        g.edata['a'] = edge_softmax(g, g.edata['s'])
        g.edata['msg'] = g.edata['V'] * g.edata['a'].unsqueeze(-1)
        g.update_all(fn.copy_e('msg', 'msg'), fn.sum('msg', 'h_new'))
        h_new = self.mlp_node(g.ndata.pop('h_new').reshape(-1, self.nh * self.d))
        h = self.norm_h(h + self.dropout_h(h_new)) if self.residual else self.norm_h(self.dropout_h(h_new))
        g.ndata['h'] = h
        g.apply_edges(lambda E: {'e_new': self.mlp_edge(torch.cat([E.src['h'], E.dst['h'], E.data['e']], -1))})
        e_new = g.edata.pop('e_new')
        e = self.norm_e(e + self.dropout_e(e_new)) if self.residual else self.norm_e(self.dropout_e(e_new))
        return h, e


class ProtMPNN(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.pre_h = nn.Linear(hidden_dim, hidden_dim)
        self.pre_e = nn.Linear(hidden_dim, hidden_dim)
        self.node_mlp = MLP(hidden_dim*2, hidden_dim, hidden_dim)
        self.edge_mlp = MLP(hidden_dim*3, hidden_dim, hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, g, h, e):
        h = self.pre_h(h)
        e = self.pre_e(e)
        g.ndata['h'] = h
        g.edata['e'] = e
        g.apply_edges(lambda E: {'msg': torch.cat([E.src['h'], E.data['e']], -1)})
        g.update_all(fn.copy_e('msg','m'), fn.sum('m','agg'))
        h_new = self.drop(self.node_mlp(g.ndata.pop('agg')))
        h = self.norm_h(h + h_new if self.residual else h_new)
        g.ndata['h'] = h
        g.apply_edges(lambda E: {'e_new': self.drop(self.edge_mlp(torch.cat([E.src['h'], E.dst['h'], E.data['e']], -1)))})
        e = self.norm_e(e + g.edata.pop('e_new') if self.residual else g.edata['e_new'])
        return h, e


class Cov2ProtMPNN(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.layer1 = ProtMPNN(hidden_dim, num_heads, dropout, residual)
        self.layer2 = ProtMPNN(hidden_dim, num_heads, dropout, residual)
    def forward(self, g, h, e):
        h, e = self.layer1(g, h, e)
        h, e = self.layer2(g, h, e)
        return h, e

class Cov2ProtGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.layer1 = ProtGraphTransformer(hidden_dim, num_heads, dropout, residual)
        self.layer2 = ProtGraphTransformer(hidden_dim, num_heads, dropout, residual)
    def forward(self, g, h, e):
        h, e = self.layer1(g, h, e)
        h, e = self.layer2(g, h, e)
        return h, e

class Seq1DCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=7, num_layers=2, dropout=0.1):
        """
        
        protein: kernel_size=7, num_layers=3
        assay: kernel_size=4, num_layers=2,
        """
        super().__init__()
        layers=[]
        for i in range(num_layers):
            input_dim=input_dim if i==0 else hidden_dim
            layers+=[
                nn.Conv1d(input_dim,hidden_dim,kernel_size,padding=(kernel_size-1)//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ]
        self.cnn=nn.Sequential(*layers)
    def forward(self,x):
        x=x.transpose(1,2)
        h=self.cnn(x)
        h=h.transpose(1,2)
        return h

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads=4,dropout=0.1):
        super().__init__()
        assert d_model%num_heads==0
        self.h=num_heads
        self.d=d_model//num_heads
        self.W_Q=nn.Linear(d_model,d_model)
        self.W_K=nn.Linear(d_model,d_model)
        self.W_V=nn.Linear(d_model,d_model)
        self.W_O=nn.Linear(d_model,d_model)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,q,k,v,k_mask=None):
        B,Lq,D=q.shape
        _,Lk,_=k.shape
        Q=self.W_Q(q)
        K=self.W_K(k)
        V=self.W_V(v)
        Q=Q.reshape(B,Lq,self.h,self.d).transpose(1,2)  # [B,h,Lq,d]
        K=K.reshape(B,Lk,self.h,self.d).transpose(1,2)  # [B,h,Lk,d]
        V=V.reshape(B,Lk,self.h,self.d).transpose(1,2)  # [B,h,Lk,d]
        attn_logits=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d)
        if k_mask is not None:
            mask=k_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
            attn_logits=attn_logits.masked_fill(~mask,float('-inf'))
        attn=F.softmax(attn_logits,dim=-1)
        attn=self.dropout(attn)
        out=torch.matmul(attn,V)  # [B,h,Lq,d]
        out=out.transpose(1,2).contiguous().reshape(B,Lq,self.h*self.d)
        out=self.W_O(out)
        q=q+self.dropout(out)
        q=self.norm1(q)
        ffn_out=self.ffn(q)
        q=q+self.dropout(ffn_out)
        q=self.norm2(q)
        return q






