# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable


    
    

    
class TeRo(nn.Module):
    def __init__(self, kg, embedding_dim, batch_size, learning_rate, L, gran, gamma, n_day, gpu=True):
        super(TeRo, self).__init__()
        self.gpu = gpu
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_day = n_day
        self.gran = gran

        self.L = L
        # Nets
        self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_R_real = torch.nn.Embedding(self.kg.n_relation*2, self.embedding_dim, padding_idx=0)
        self.emb_R_img = torch.nn.Embedding(self.kg.n_relation*2, self.embedding_dim, padding_idx=0)
        self.emb_Time = torch.nn.Embedding(n_day, self.embedding_dim, padding_idx=0)
        
        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.emb_E_real.weight.data.uniform_(-r, r)
        self.emb_E_img.weight.data.uniform_(-r, r)
        self.emb_R_real.weight.data.uniform_(-r, r)
        self.emb_R_img.weight.data.uniform_(-r, r)
        self.emb_Time.weight.data.uniform_(-r, r)
        # self.emb_T_img.weight.data.uniform_(-r, r)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        
        if self.gpu:
            self.cuda()



    def forward(self, X):
        h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.int64)//self.gran

        if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            r_i = Variable(torch.from_numpy(r_i).cuda())
            d_i = Variable(torch.from_numpy(d_i).cuda())
        else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            r_i = Variable(torch.from_numpy(r_i))
            d_i = Variable(torch.from_numpy(d_i))

        pi = 3.14159265358979323846
        d_img = torch.sin(self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(
            self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_real-\
                 self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_img

        t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_real-\
                 self.emb_E_img(t_i).view(-1,self.embedding_dim)*d_img


        r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)

        h_img = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_img+\
                 self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_real


        t_img = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_img+\
                self.emb_E_img(t_i).view(-1,self.embedding_dim) *d_real

        r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)



        if self.L == 'L1':
            out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
            out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
            out = out_real + out_img

        else:
            out_real = torch.sum((h_real + r_real + d_i - t_real) ** 2, 1)
            out_img = torch.sum((h_img + r_img + d_i + t_real) ** 2, 1)
            out = torch.sqrt(out_img + out_real)

        return out
