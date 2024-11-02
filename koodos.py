# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 19:35
@Author  : Zekun Cai
@File    : koodos.py
@Software: PyCharm
'''
import torch
import torch.nn as nn

from torchdiffeq import odeint

dataset_modules = {
    'Moons': 'model.moon',
    'MNIST': 'model.mnist',
    'YearBook': 'model.yearbook',
    'Twitter': 'model.twitter',
    'Cyclone': 'model.cyclone',
    'House': 'model.house'
}


class Koodos(nn.Module):
    def __init__(self, dataset, time_point):
        super(Koodos, self).__init__()

        if dataset in dataset_modules:
            module = __import__(dataset_modules[dataset], fromlist=['backbone_model', 'predictive_model', 'generalized_model',
                                                                    'encoder', 'decoder', 'dyna_f', 'ode_method', 'rk_step'])
        else:
            raise ValueError('Not defined dataset!')

        self.time_point = time_point
        self.shared_model = module.backbone_model
        self.pred_model = module.predictive_model
        self.gene_model = module.generalized_model
        self.encoder = module.encoder
        self.decoder = module.decoder
        self.dynamic = module.dyna_f()
        self.method = module.ode_method
        self.step = module.rk_step

    # Obtain prediction and predictive model parameters from each prediction model
    def predictive_model_pred(self, X, idx):
        seg_length = len(X)
        pred, param = [], []
        for i in range(seg_length):
            x = X[i]
            pred.append(self.pred_model[idx + i](x))
            param.append(torch.cat([p.flatten() for p in self.pred_model[idx + i].parameters()]))
        param = torch.stack(param)
        return param, pred

    # Obtain prediction from each generalized model
    def generalized_model_pred(self, X, Gene_Param):
        seg_length = Gene_Param.shape[0]
        pred = []
        for d in range(seg_length):
            x, param = X[d], Gene_Param[d]
            y = self.gene_model(x, param)
            pred.append(y)
        return pred

    def forward(self, X, continous_time=None, idx=0):
        '''
        :param X: The sequence segment of temporal domain data -> [Di, Di+1, ..., Di+seg_len]
        :param continous_time: Timestamp corresponding to the domain -> [ti, ti+1, ..., ti+seg_len]
        :param idx: Starting index of the current domain sequence segment -> i
        :return: Prediction and state for predictive and generalized models
        '''

        X = [self.shared_model(x) for x in X]
        init_param, init_pred = self.predictive_model_pred(X, idx)  # theta_i and predictive model prediction
        init_embed = self.encoder(init_param)  # z_i
        gene_embed = odeint(self.dynamic, init_embed[0], continous_time, method=self.method, options={'step_size': self.step})  # z^{j \rightarrow i}_i
        gene_param = self.decoder(gene_embed)  # \theta^{j \rightarrow i}_i
        init_debed = self.decoder(init_embed)  # \varphi^{-1}(\varphi(\theta_i))
        gene_pred = self.generalized_model_pred(X, gene_param)  # generalized model prediction
        return init_pred, gene_pred, init_param, init_embed, init_debed, gene_param, gene_embed
