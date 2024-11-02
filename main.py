# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 19:35
@Author  : Zekun Cai
@File    : main.py
@Software: PyCharm
'''
import sys
import shutil
import argparse
from datetime import datetime

from koodos import *
from util import *
from param import *


def trainModel(X_train, Y_train, time_train):
    # To avoid time consumption and cumulative errors in integrating long sequences,
    # we divide the domain sequence into segments by sliding window with no longer than seg_len and integrate each segment separately.
    train_domain_seg = []
    for idx in range(n_train - 1):
        l = min(n_train - idx, seg_len)
        seg_time, seg_X, seg_Y = time_train[idx:idx + l], X_train[idx:idx + l], Y_train[idx:idx + l]
        train_domain_seg.append([seg_time, seg_X, seg_Y])

    # The last few segments are used for validation
    n_seg, n_train_seg = len(train_domain_seg), len(train_domain_seg) - n_val_seg
    idx_train_seg = np.arange(n_train_seg, dtype=np.int64)
    idx_val_seg = np.arange(n_train_seg, n_seg, dtype=np.int64)

    # Initialization model-related
    model = Koodos(data_set, time_train).to(device)
    optimizer = torch.optim.Adam([{'params': model.shared_model.parameters(), 'lr': pred_learning_rate, 'weight_decay': weight_decay},
                                  {'params': model.pred_model.parameters(), 'lr': pred_learning_rate, 'weight_decay': weight_decay},
                                  {'params': model.encoder.parameters(), 'lr': coder_learning_rate, 'weight_decay': weight_decay},
                                  {'params': model.decoder.parameters(), 'lr': coder_learning_rate, 'weight_decay': weight_decay},
                                  {'params': model.dynamic.parameters(), 'lr': dyn_learning_rate, 'weight_decay': weight_decay}])

    # Training
    min_val_loss = np.inf
    model.train()
    for e in range(epoch):
        epoch_loss = 0

        for batch_idx in np.array_split(idx_train_seg, n_train_seg // batch):
            loss = 0
            for idx in batch_idx:
                seg_time, seg_X, seg_Y = train_domain_seg[idx]
                init_pred, gene_pred, init_param, init_embed, init_debed, gene_param, gene_embed = model(seg_X, seg_time, idx)

                # Calculate the loss
                loss_intri, loss_integ = get_task_loss(data_set, seg_Y, init_pred, gene_pred)
                loss_recon = F.mse_loss(init_param, init_debed)
                loss_dyna = F.mse_loss(init_embed, gene_embed)
                loss_consis = F.mse_loss(init_param, gene_param)
                loss = loss + alpha * loss_intri + alpha * loss_integ + beta * loss_recon + gamma * loss_dyna + beta * loss_consis

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate generalization on the last few domain sequences
        val_loss = 0
        for idx in idx_val_seg:
            seg_time, seg_X, seg_Y = train_domain_seg[idx]
            init_pred, gene_pred, _, _, _, _, _ = model(seg_X, seg_time, idx)
            _, loss_integ = get_task_loss(data_set, seg_Y, init_pred, gene_pred)
            val_loss = val_loss + loss_integ.item()

        if val_loss < min_val_loss:
            # For saving IO time
            if e > epoch * 0.8:
                min_val_loss = val_loss
                torch.save(model.state_dict(), save_path + '/model.pt')

        # Log
        print('Epoch: {}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(e, epoch_loss, val_loss))
        with open(save_path + '/log.txt', 'a') as f:
            f.write("{},{},{}\n".format(e, epoch_loss, val_loss))

    model.load_state_dict(torch.load(save_path + '/model.pt'.format(data_set)))
    return model


def testModel(model, X_text, Y_test, time_test):
    model.eval()
    with torch.no_grad():
        last_param = torch.cat([p.flatten() for p in model.pred_model[-1].parameters()])
        last_time = model.time_point[-1:]
        final_time_test = torch.cat([last_time, time_test])

        last_embed = model.encoder(last_param)
        test_embed = odeint(model.dynamic, last_embed, final_time_test, method=model.method, options={'step_size': model.step})[1:]
        test_param = model.decoder(test_embed)
        X_text = [model.shared_model(x) for x in X_text]
        test_pred = model.generalized_model_pred(X_text, test_param)

    # Test the performance
    score = get_task_score(data_set, torch.cat(Y_test), torch.cat(test_pred))
    f = open(save_path + '/scores.txt', 'a')
    print('\nAll Metric {:.4}\n'.format(score))
    f.write('All Metric {}\n'.format(score))

    for i in range(len(Y_test)):
        Y_test_step, test_pred_step = Y_test[i], test_pred[i]
        score = get_task_score(data_set, Y_test_step, test_pred_step)
        print('Step {}, Metric {:.4}'.format(i + 1, score))
        f.write('Step {}, Metric {}\n'.format(i + 1, score))


def main():
    X, Y, time_points = dataset_preparation(data_set, data_path, device)
    X_train, Y_train, time_train, = X[:n_train], Y[:n_train], time_points[:n_train]
    X_text, Y_test, time_test = X[n_train:], Y[n_train:], time_points[n_train:]

    model = trainModel(X_train, Y_train, time_train)
    testModel(model, X_text, Y_test, time_test)


# Parameters Loading
##################################################################################################
parser = argparse.ArgumentParser(description='Set dataset and CUDA device.')
parser.add_argument('--dataset', type=str, default='Moons', help='Name of the dataset.')
parser.add_argument('--cuda', type=int, default=1, help='CUDA device number.')
args = parser.parse_args()

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
data_set = args.dataset

data_setting = SETTING[data_set]
data_path = data_setting['data_file']

n_train = data_setting['n_train']
seg_len = data_setting['seg_len']
n_val_seg = data_setting['n_val_seg']

epoch = data_setting['epoch']
batch = data_setting['batch']
pred_learning_rate = data_setting['pred_learning_rate']
coder_learning_rate = data_setting['coder_learning_rate']
dyn_learning_rate = data_setting['dyn_learning_rate']
weight_decay = data_setting['weight_decay']

alpha = data_setting['alpha']
beta = data_setting['beta']
gamma = data_setting['gamma']

keyword = 'pred_' + data_set + '_' + datetime.now().strftime("%y%m%d%H%M%S")
save_path = './save/' + keyword
##################################################################################################

if __name__ == '__main__':
    currentPython = sys.argv[0]
    shutil.copytree('model', save_path + '/model', dirs_exist_ok=True)
    shutil.copy2('koodos.py', save_path)
    shutil.copy2('param.py', save_path)
    shutil.copy2('util.py', save_path)
    shutil.copy2(currentPython, save_path)

    main()
