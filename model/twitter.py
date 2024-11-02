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

data_dim = SETTING['Twitter']['data_dim']
gene_dim = SETTING['Twitter']['gene_dim']
embed_dim = SETTING['Twitter']['embed_dim']
n_train = SETTING['Twitter']['n_train']
ode_method = SETTING['Twitter']['ode_method']
rk_step = SETTING['Twitter']['rk_step']


# The first feature of the dataset is a categorical feature (State No.)
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


# Define the Generalized Model, it receives the domain data x and the generalized model parameters,
# then computes the domain data label y.
# The model structure is the same as the predictive model.
def generalized_model(domain_x, domain_param):
    # Build parameter structures for generalized model
    weights = {}
    biases = {}
    start_idx = 0
    for name, p in predictive_model[0].state_dict().items():
        end_idx = start_idx + p.numel()
        if name.endswith("bias"):
            biases[name] = domain_param[start_idx:end_idx].view(p.shape)
        elif name.endswith("weight"):
            weights[name] = domain_param[start_idx:end_idx].view(p.shape)
        else:
            raise ValueError('Not defined layer!')
        start_idx = end_idx

    # Forward pass through the generalized parameters
    x = domain_x
    for name, layer in predictive_model[0].named_children():
        if isinstance(layer, nn.Linear):
            weight_name = f"{name}.weight"
            bias_name = f"{name}.bias"
            x = F.linear(x, weights[weight_name], biases[bias_name])
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)
        elif isinstance(layer, nn.Sigmoid):
            x = torch.sigmoid(x)
        elif isinstance(layer, nn.LogSoftmax):
            x = F.log_softmax(x, dim=layer.dim)
        elif isinstance(layer, nn.Dropout):
            pass  # Dropout is ignored during manual computation
        else:
            raise ValueError('Not defined layer!')

    domain_y = x
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
class dyna_f(nn.Module):
    def __init__(self):
        super(dyna_f, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, t, y):
        return self.net(y)
