# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 21:32
@Author  : Zekun Cai
@File    : moon.py
@Software: PyCharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from param import *

data_dim = SETTING['Moons']['data_dim']      # the dimension of the data
gene_dim = SETTING['Moons']['gene_dim']      # the dimension of the generalized model
embed_dim = SETTING['Moons']['embed_dim']    # the dimension of the Koopman Space
n_train = SETTING['Moons']['n_train']
ode_method = SETTING['Moons']['ode_method']
rk_step = SETTING['Moons']['rk_step']

backbone_model = nn.Sequential()

# Define the Predictive Model, each training domain has a corresponding predictive model
predictive_model = nn.ModuleList([nn.Sequential(
            nn.Linear(data_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()) for i in range(n_train)])

# Define the Generalized Model, it receives the domain data X and the generalized model parameters, then computes the domain data label Y.
# The model structure is the same as the predictive model.
def generalized_model(domain_x, domain_param):
    m_1 = domain_param[:data_dim * 50]
    b_1 = domain_param[data_dim * 50:data_dim * 50 + 50]
    m_2 = domain_param[data_dim * 50 + 50:data_dim * 50 + 50 + 50 * 50]
    b_2 = domain_param[data_dim * 50 + 50 + 50 * 50:data_dim * 50 + 50 + 50 * 50 + 50]
    m_3 = domain_param[data_dim * 50 + 50 + 50 * 50 + 50:data_dim * 50 + 50 + 50 * 50 + 50 + 50]
    b_3 = domain_param[-1]

    domain_y = torch.sigmoid(F.linear(torch.relu(F.linear(torch.relu(F.linear(domain_x, m_1.reshape((50, data_dim)), b_1)), m_2.reshape((50, 50)), b_2)),m_3.reshape((1, 50)), b_3))
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
        self.net = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False))

    def forward(self, t, y):
        return self.net(y)

# # Define the Koopman Operator with priori knowledge
# class ode_f(nn.Module):
#     def __init__(self):
#         super(ode_f, self).__init__()
#         self.L = nn.Parameter(torch.randn((embed_dim, embed_dim)), requires_grad=True)

#     def forward(self, t, y):
#         K = self.L - self.L.T
#         return torch.matmul(y, K)