# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 21:32
@Author  : Zekun Cai
@File    : cyclone.py
@Software: PyCharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from param import *

data_dim = SETTING['Cyclone']['data_dim']      # the dimension of the data
gene_dim = SETTING['Cyclone']['gene_dim']     # the dimension of the generalized model
embed_dim = SETTING['Cyclone']['embed_dim']    # the dimension of the Koopman Space
n_train = SETTING['Cyclone']['n_train']
ode_method = SETTING['Cyclone']['ode_method']
rk_step = SETTING['Cyclone']['rk_step']

backbone_model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.7),
        )

# Define the Predictive Model, each training domain has a corresponding predictive model
predictive_model = nn.ModuleList([nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 1)) for i in range(n_train)])

# Define the Generalized Model, it receives the domain data X and the generalized model parameters, then computes the domain data label Y.
# The model structure is the same as the predictive model.
def generalized_model(domain_x, domain_param):
    weights = {}
    biases = {}
    start_idx = 0
    for name, p in predictive_model[0].state_dict().items():
        end_idx = start_idx + p.numel()
        if 'bias' in name:
            biases[name] = domain_param[start_idx:end_idx].view(p.shape)
        else:
            weights[name] = domain_param[start_idx:end_idx].view(p.shape)
        start_idx = end_idx

    domain_y = F.linear(F.relu(F.linear(F.relu(F.linear(domain_x, weights['0.weight'], biases['0.bias'])), weights['3.weight'], biases['3.bias'])), weights['6.weight'], biases['6.bias'])
    return domain_y

# Define the Encoder and the Decoder
encoder = nn.Sequential(
            nn.Linear(gene_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim))

decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, gene_dim))

# Define the Koopman Operator
class ode_f(nn.Module):
    def __init__(self):
        super(ode_f, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, t, y):
        return self.net(y)
