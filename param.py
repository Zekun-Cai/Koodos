# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 21:14
@Author  : Zekun Cai
@File    : param.py
@Software: PyCharm
'''
SETTING = {
    'Moons': {
        'name': '2-Moons',
        'data_file': '/home/cai/code/Continuous_Domain/code/data/CTD/Moons50/dataset.pkl',
        'data_dim': 2,
        'data_size': 1000,
        'n_domains': 50,
        'task': 'binary_classification',

        'n_train': 35,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 250,
        'batch': 10,
        'pred_learning_rate': 1e-2,
        'coder_learning_rate': 1e-3,
        'ode_learning_rate': 1e-3,

        'alpha': 1,
        'beta': 100,
        'gamma': 10,

        'gene_dim': 2751,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,
    },

    'MNIST': {
        'name': 'MNIST',
        'data_file': '/home/cai/code/Continuous_Domain/code/data/CTD/MNIST50/dataset.pkl',
        'data_dim': 28,
        'data_size': 1000,
        'n_domains': 50,
        'task': 'classification',

        'n_train': 35,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 300,
        'batch': 2,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'ode_learning_rate': 1e-3,

        'alpha': 1,
        'beta': 100,
        'gamma': 10,

        'gene_dim': 75146,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,
    },

    'YearBook': {
        'name': 'YearBook',
        'data_file': '/home/cai/code/Continuous_Domain/code/data/CTD/YearBook/dataset.pkl',
        'data_dim': 32,
        'data_size': None,
        'n_domains': 40,
        'task': 'binary_classification',

        'n_train': 28,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 200,
        'batch': 1,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'ode_learning_rate': 1e-3,

        'alpha': 0.1,
        'beta': 100,
        'gamma': 10,

        'gene_dim': 135361,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,
    },

    'Twitter': {
        'name': 'Twitter',
        'data_file': '/home/cai/code/Continuous_Domain/code/github/data/CTD/Influenza/dataset.pkl',
        'data_dim': 526,
        'data_size': None,
        'n_domains': 50,
        'task': 'binary_classification',

        'n_train': 35,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 200,
        'batch': 1,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'ode_learning_rate': 1e-3,

        'alpha': 1,
        'beta': 10,
        'gamma': 10,

        'gene_dim': 8385,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,
    },

    'Cyclone': {
        'name': 'Cyclone',
        'data_file': '/home/cai/code/Continuous_Domain/code/github/data/CTD/TCIR/dataset_2016_d.pkl',
        'data_dim': 64,
        'data_size': None,
        'n_domains': 72,
        'task': 'regression',

        'n_train': 50,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 200,
        'batch': 1,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'ode_learning_rate': 1e-3,

        'alpha': 1,
        'beta': 100,
        'gamma': 100,

        'gene_dim': 135361,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,
    },

    'House': {
        'name': 'House',
        'data_file': '/home/cai/code/Continuous_Domain/code/github/data/CTD/HousePrice/dataset.pkl',
        'data_dim': 30,
        'data_size': None,
        'n_domains': 40,
        'task': 'regression',

        'n_train': 28,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 200,
        'batch': 1,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'ode_learning_rate': 1e-3,

        'alpha': 1,
        'beta': 10,
        'gamma': 10,

        'gene_dim': 173201,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,
    },
}