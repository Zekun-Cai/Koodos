# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 19:35
@Author  : Zekun Cai
@File    : util.py
@Software: PyCharm
'''
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score


def load_datasets(filename="dataset.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["datasets"], data["time_points"]


def dataset_preparation(dataset, path, device):
    datasets, time_points = load_datasets(path)

    if dataset == 'Moons':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'MNIST':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [torch.Tensor(item[1]).long().to(device) for item in datasets]
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'YearBook':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        # Scale the timeline to match the rk step size.
        # We set the maximum time to 50 so that it is not too large (too many integral steps) or too small (not enough precision).
        # Note that this value does not must be set to 50.
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Twitter':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Cyclone':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'House':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = time_points / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)
    else:
        raise ValueError('Not defined dataset!')

    return X, Y, time_points


def get_task_loss(dataset, Y, init_pred, gene_pred):
    init_pred, gene_pred, Y = torch.cat(init_pred), torch.cat(gene_pred), torch.cat(Y)

    if dataset in ['Moons', 'YearBook', 'Twitter']:
        loss_intri = F.binary_cross_entropy(init_pred, Y)
        loss_integ = F.binary_cross_entropy(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset in ['MNIST']:
        loss_intri = F.nll_loss(init_pred, Y)
        loss_integ = F.nll_loss(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset in ['Cyclone', 'House']:
        loss_intri = F.l1_loss(init_pred, Y)
        loss_integ = F.l1_loss(gene_pred, Y)
        return loss_intri, loss_integ

    else:
        raise ValueError('Not defined dataset!')


def get_task_score(dataset, Y, pred):
    if dataset in ['Moons', 'YearBook']:
        prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()
        accuracy = accuracy_score(Y.cpu().numpy(), prediction.cpu().numpy())
        return accuracy

    elif dataset in ['MNIST']:
        prediction = pred.argmax(dim=1)
        accuracy = accuracy_score(Y.view(-1).cpu().numpy(), prediction.cpu().numpy())
        return accuracy

    elif dataset in ['Twitter']:
        prediction = torch.as_tensor((pred.detach() - 0.5) > 0).float()
        accuracy = accuracy_score(Y.cpu().numpy(), prediction.cpu().numpy())
        auc = roc_auc_score(Y.cpu().numpy(), pred.detach().cpu().numpy(), labels=[0, 1])
        return auc

    elif dataset in ['Cyclone', 'House']:
        mae = F.l1_loss(Y, pred).item()
        return mae

    else:
        raise ValueError('Not defined dataset!')
