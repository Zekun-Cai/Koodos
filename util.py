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
        # Scaling the timeline to match the rk step size.
        # The maximum value does not have to be 50, but it should not be too large (long computation times) or too small (insufficient precision).
        time_points = np.array(time_points) / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Twitter':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = np.array(time_points) / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'Cyclone':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        X = [torch.transpose(x, 1, 3) for x in X]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = np.array(time_points) / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device)

    elif dataset == 'House':
        X = [torch.Tensor(item[0]).to(device) for item in datasets]
        Y = [torch.Tensor(item[1])[:, None].to(device) for item in datasets]
        time_points = np.array(time_points) / np.max(time_points) * 50
        time_points = torch.Tensor(time_points).to(device) #修改数据格式
    else:
        raise ValueError('Not defined dataset!')

    return X, Y, time_points


def get_task_loss(dataset, Y, init_pred, gene_pred):
    init_pred, gene_pred, Y = torch.cat(init_pred), torch.cat(gene_pred), torch.cat(Y)

    if dataset == 'Moons':
        loss_intri = F.binary_cross_entropy(init_pred, Y)
        loss_integ = F.binary_cross_entropy(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset == 'MNIST':
        loss_intri = F.nll_loss(init_pred, Y)
        loss_integ = F.nll_loss(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset == 'YearBook':
        loss_intri = F.binary_cross_entropy(init_pred, Y)
        loss_integ = F.binary_cross_entropy(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset == 'Twitter':
        loss_intri = F.binary_cross_entropy(init_pred, Y)
        loss_integ = F.binary_cross_entropy(gene_pred, Y)
        return loss_intri, loss_integ

    elif dataset == 'Cyclone' or 'House':
        loss_intri = F.l1_loss(init_pred, Y)
        loss_integ = F.l1_loss(gene_pred, Y)
        return loss_intri, loss_integ

    else:
        raise ValueError('Not defined dataset!')


def get_task_score(dataset, Y_test, test_pred):
    if dataset == 'Moons':
        prediction = torch.as_tensor((test_pred.detach() - 0.5) > 0).float()
        accuracy = accuracy_score(Y_test.cpu().numpy(), prediction.cpu().numpy())
        return accuracy

    elif dataset == 'MNIST':
        prediction = test_pred.argmax(dim=1)
        accuracy = accuracy_score(Y_test.view(-1).cpu().numpy(), prediction.cpu().numpy())
        return accuracy

    elif dataset == 'YearBook':
        prediction = torch.as_tensor((test_pred.detach() - 0.5) > 0).float()
        accuracy = accuracy_score(Y_test.cpu().numpy(), prediction.cpu().numpy())
        return accuracy

    elif dataset == 'Twitter':
        prediction = torch.as_tensor((test_pred.detach() - 0.5) > 0).float()
        accuracy = accuracy_score(Y_test.cpu().numpy(), prediction.cpu().numpy())
        auc = roc_auc_score(Y_test.cpu().numpy(), test_pred.detach().cpu().numpy(), labels=[0, 1])
        return auc

    elif dataset == 'Cyclone' or 'House':
        mae = F.l1_loss(Y_test, test_pred).item()
        return mae
    else:
        raise ValueError('Not defined dataset!')
