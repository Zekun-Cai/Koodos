# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 21:32
@Author  : Zekun Cai
@File    : twitter.py
@Software: PyCharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from param import *

data_dim = SETTING['Twitter']['data_dim']      # the dimension of the data
gene_dim = SETTING['Twitter']['gene_dim']     # the dimension of the generalized model
embed_dim = SETTING['Twitter']['embed_dim']    # the dimension of the Koopman Space
n_train = SETTING['Twitter']['n_train']
ode_method = SETTING['Twitter']['ode_method']
rk_step = SETTING['Twitter']['rk_step']

class EmbeddingNet(nn.Module):
    def __init__(self, discrete_value=57, embedding_dim=10):
        super(EmbeddingNet, self).__init__()
        self.embedding = nn.Embedding(discrete_value, embedding_dim)
        self.fc1 = nn.Linear(data_dim - 1 + embedding_dim, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        discrete_feature = x[:, 0].long()
        continuous_features = x[:, 1:]
        embedded_feature = self.embedding(discrete_feature)
        x = torch.cat((embedded_feature, continuous_features), dim=1)
        x = self.relu(self.fc1(x))
        return x

backbone_model = EmbeddingNet()

# Define the Predictive Model, each training domain has a corresponding predictive model
predictive_model = nn.ModuleList([nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()) for i in range(n_train)])

# Define the Generalized Model, it receives the domain data X and the generalized model parameters, then computes the domain data label Y.
# The model structure is the same as the predictive model.
def generalized_model(domain_x, domain_param):
    m_1 = domain_param[:32 * 128]
    b_1 = domain_param[32 * 128:32 * 128 + 128]
    m_2 = domain_param[32 * 128 + 128:32 * 128 + 128 + 128 * 32]
    b_2 = domain_param[32 * 128 + 128 + 128 * 32:32 * 128 + 128 + 128 * 32 + 32]
    m_3 = domain_param[32 * 128 + 128 + 128 * 32 + 32:32 * 128 + 128 + 128 * 32 + 32 + 32]
    b_3 = domain_param[-1]

    domain_y = torch.sigmoid(F.linear(torch.relu(F.linear(torch.relu(F.linear(domain_x, m_1.reshape((128, 32)), b_1)), m_2.reshape((32, 128)), b_2)), m_3.reshape((1, 32)), b_3))

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
