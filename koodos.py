# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 19:35
@Author  : Zekun Cai
@File    : koodos.py
@Software: PyCharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

class Koodos(nn.Module):
    def __init__(self, dataset, time_point):
        super(Koodos, self).__init__()
        self.time_point = time_point

        if dataset == 'Moons':
            from model.moon import backbone_model, predictive_model, generalized_model, encoder, decoder, ode_f, ode_method, rk_step
        elif dataset == 'MNIST':
            from model.mnist import backbone_model, predictive_model, generalized_model, encoder, decoder, ode_f, ode_method, rk_step
        elif dataset == 'YearBook':
            from model.yearbook import backbone_model, predictive_model, generalized_model, encoder, decoder, ode_f, ode_method, rk_step
        elif dataset == 'Twitter':
            from model.twitter import backbone_model, predictive_model, generalized_model, encoder, decoder, ode_f, ode_method, rk_step
        elif dataset == 'Cyclone':
            from model.cyclone import backbone_model, predictive_model, generalized_model, encoder, decoder, ode_f, ode_method, rk_step
        elif dataset == 'House':
            from model.house import backbone_model, predictive_model, generalized_model, encoder, decoder, ode_f, ode_method, rk_step
        else:
            raise ValueError('Not define dataset!')
        self.shared_model = backbone_model
        self.pred_model = predictive_model
        self.gene_model = generalized_model
        self.encoder = encoder
        self.decoder = decoder
        self.odefunc = ode_f()
        self.method = ode_method
        self.step = rk_step

    # For getting the prediction and parameters from the predictive model
    def predictive_model_pred(self, X, idx):
        seg_length = len(X)
        pred, param = [], []
        for i in range(seg_length):
            x = X[i]
            pred.append(self.pred_model[idx + i](x))
            param.append(torch.cat([p.flatten() for p in self.pred_model[idx + i].parameters()]))
        param = torch.stack(param)
        return param, pred

    # For getting the prediction from the generalized model
    def generalized_model_pred(self, X, Gene_Param):
        seg_length = Gene_Param.shape[0]
        pred = []
        for d in range(seg_length):
            x, param = X[d], Gene_Param[d]
            y = self.gene_model(x, param)
            pred.append(y)
        return pred

    def forward(self, X, continous_time=None, idx=0):
        X = [self.shared_model(x) for x in X]
        init_param, init_pred = self.predictive_model_pred(X, idx)  # theta_j and predictive model prediction
        init_embed = self.encoder(init_param)  # z_j
        gene_embed = odeint(self.odefunc, init_embed[0], continous_time, method=self.method, options={'step_size': self.step})  # z^{j \rightarrow i}_i
        gene_param = self.decoder(gene_embed)  # \theta^{j \rightarrow i}_i
        init_debed = self.decoder(init_embed)  # \varphi^{-1}(\varphi(\theta_j))
        gene_pred = self.generalized_model_pred(X, gene_param)  # generalized model prediction
        return init_pred, gene_pred, init_param, init_embed, init_debed, gene_param, gene_embed