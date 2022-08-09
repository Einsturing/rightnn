import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
from scatterFig import figsave, display, data_to_img
import os
# from gan_ddp import GRUNet, Generator, Generator_rcc, Discriminator, split_sequence, MoveDataset, SpatialDiscriminator, DBlock, TemporalDiscriminator, Discriminator_Line
from resnet_test import GRUNet, Generator, Generator_rcc, Discriminator, Discriminator_Line, split_sequence, MoveDataset, GModel
# from mgmfnlo2 import Model, LSTMNet, split_sequence, MoveDataset
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse


def load_m():
    file_name = os.path.join('7caddld1499.pt')
    with open(file_name, 'rb') as f:
        model = torch.load(f, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu', 'cuda:4': 'cpu', 'cuda:5': 'cpu', 'cuda:6': 'cpu', 'cuda:7': 'cpu'})
        # model = torch.load(f).module.cpu()
    return model


n_steps = 4
multi = 6
jump = 1

# radar_tmp = pd.read_csv('radar_tmp.csv')
# tmpX = radar_tmp.dropna().x
# tmpY = radar_tmp.y
# lenX = len(tmpX)
# lenY = len(tmpY)
# npX = np.reshape(np.array([[tmpX[i] for j in range(113, 241)] for i in range(96, 224)]), (-1, 1))
# npY = np.reshape(np.array([[tmpY[j] for j in range(113, 241)] for i in range(96, 224)]), (-1, 1))
# X = pd.DataFrame(npX)
# Y = pd.DataFrame(npY)

# full = pd.read_csv('uk190624-256.csv')
df180101 = pd.read_csv('uk180101.csv').fillna(0).iloc[:, 2:].T.values
df180102 = pd.read_csv('uk180102.csv').fillna(0).iloc[:, 2:].T.values
df190628 = pd.read_csv('uk190628.csv').fillna(0).iloc[:, 2:].T.values
df190629 = pd.read_csv('uk190629.csv').fillna(0).iloc[:, 2:].T.values
df190630 = pd.read_csv('uk190630.csv').fillna(0).iloc[:, 2:].T.values
df220101 = pd.read_csv('uk220101.csv').fillna(0).iloc[:, 2:].T.values
df190624 = pd.read_csv('uk190624.csv').fillna(0).iloc[:, 2:].T.values
# for i in range(len(df190624)):
#     data_to_img(df190624[i].reshape(1, -1), '19/image{}'.format(i + 1), 256, 256)
# df = pd.read_csv('uk190624.csv').fillna(0).iloc[:, 2:].T.values
# df = np.concatenate((df180101, df180102, df190628, df190629, df190630, df220101, df190624), axis=0)
# df = pd.read_csv('uk190629.csv').fillna(0).iloc[:, 2:].T.values
# df[np.where(df > 8.0)] = 8.0
scaler = MinMaxScaler(feature_range=(0, 1))
# df = scaler.fit_transform(df)
df = scaler.fit_transform(
    np.concatenate((df180101, df180102, df190628, df190629, df190630, df220101, df190624), axis=0))
data = torch.Tensor(df).T
grid_shape = int(math.sqrt(data.shape[0]))
timestep = data.shape[1]
# draw = pd.read_csv('once12-gt-pred-6071.csv')
# draw.insert(0, 'y', full.y)
# draw.insert(0, 'x', full.x)
# figsave(draw, draw.columns[1 + 2], draw.columns[25 + 2], v=0.4, name='once12-71p-59gt.png')
# figsave(draw, draw.columns[1 + 2], draw.columns[13 + 2], v=0.4, name='once12-71gt-59gt.png')
# figsave(draw, draw.columns[1 + 2], draw.columns[16 + 2], v=0.4, name='once12-62p-59gt.png')
# figsave(draw, draw.columns[1 + 2], draw.columns[19 + 2], v=0.4, name='once12-65p-59gt.png')
# figsave(draw, draw.columns[1 + 2], draw.columns[22 + 2], v=0.4, name='once12-68p-59gt.png')
# figsave(draw, draw.columns[1 + 2], draw.columns[25 + 2], v=0.4, name='once12-71p-59gt.png')

# move = pd.read_csv('pst_full.csv')
# df = pd.read_csv('pst.csv').to_numpy()
# data = torch.Tensor(df)
inp = data[:, 0].reshape(grid_shape, grid_shape).unsqueeze(0)
for i in range(1, timestep):
    tmp = data[:, i].reshape(grid_shape, grid_shape).unsqueeze(0)  # C2D + FC, C1D, LSTM
    #     # tmp = data[:, i].unsqueeze(0)  # FC, Conv1D, LSTM
    inp = torch.cat((inp, tmp), 0)

# inp = inp[:, 96:224, 113:241]
# slice = int(timestep * 0.8)
# test_set = inp[slice - (n_steps + multi - 1):]
slice = 288 * 6 + 220
test_set = inp[slice - 4:]
# test_set = inp[slice - n_steps:]
test_x, test_y = split_sequence(test_set, n_steps=n_steps, multi=multi)
test = MoveDataset(test_x, test_y)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
# dist.init_process_group("nccl", init_method='env://')
model = load_m()

# draw_set = inp[60:64].unsqueeze(1)
# conv_draw_set, tconv_draw_set = model.test(draw_set)
# draw_set = scaler.inverse_transform(draw_set.reshape(4, -1))
# conv_draw_set = scaler.inverse_transform(conv_draw_set.detach().reshape(4, -1))
# tconv_draw_set = scaler.inverse_transform(tconv_draw_set.detach().reshape(4, -1))
# dfdraw = pd.DataFrame(draw_set.T)
# dfconvdraw = pd.DataFrame(conv_draw_set.T)
# dftconvdraw = pd.DataFrame(tconv_draw_set.T)
# X = pd.DataFrame(torch.Tensor([[j for j in range(256)] for i in range(256)]).view(256 ** 2).numpy())
# Y = pd.DataFrame(torch.Tensor([[i for j in range(256)] for i in range(256)]).view(256 ** 2).numpy())
# dfdraw.insert(0, 'y', Y)
# dfdraw.insert(0, 'x', X)
# dfconvdraw.insert(0, 'y', Y)
# dfconvdraw.insert(0, 'x', X)
# dftconvdraw.insert(0, 'y', Y)
# dftconvdraw.insert(0, 'x', X)
# for i in range(2, 6):
#     data_to_img(dfdraw.iloc[:, i], 'draw/dti_{}'.format(i))
#     display(dfdraw, dfdraw.columns[i], name='draw/scat_')
#
#     data_to_img(dfconvdraw.iloc[:, i], 'draw/dti_conv_{}'.format(i))
#     display(dfconvdraw, dfconvdraw.columns[i], name='draw/scat_conv_')
#
#     data_to_img(dftconvdraw.iloc[:, i], 'draw/dti_tconv_{}'.format(i))
#     display(dftconvdraw, dftconvdraw.columns[i], name='draw/scat_tconv_')

# groundtruth37 = inp[37].numpy().reshape(1, -1)
# groundtruth59 = inp[39 - (n_steps + multi - 1)].numpy().reshape(1, -1)
# groundtruth59 = scaler.inverse_transform(groundtruth59).T
# dfg59 = pd.DataFrame(groundtruth59)
# dfg59.to_csv('59.csv')
# groundtruth63 = inp[63].numpy().reshape(1, -1)
# groundtruth63 = scaler.inverse_transform(groundtruth63).T
# dfg63 = pd.DataFrame(groundtruth63)
# gt6071 = inp[60:].numpy().reshape(12, -1)
# gt6071 = scaler.inverse_transform(gt6071).T
# gt6071 = pd.DataFrame(gt6071)
# gt6071.to_csv('gt6071.csv')

model.eval()
with torch.no_grad():
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param)

    step = 1
    for idx, (inputs, tlabels) in enumerate(test_loader):
        inputs = inputs.float().permute(1, 0, 2, 3)
        tlabels = tlabels.float().permute(1, 0, 2, 3)
        # lpreds = model(inputs)
        lpreds = model.predict(inputs)
        # lpreds_test = lpreds.reshape(multi, -1).cpu().numpy().T
        # gt_test = tlabels.reshape(multi, -1).cpu().numpy().T
        lprediction = scaler.inverse_transform(lpreds.reshape(1, -1).cpu())
        gt = scaler.inverse_transform(tlabels[-1].reshape(1, -1).cpu())
        loss = math.sqrt(mean_squared_error(lprediction, gt))
        print("rmse:", loss)
        dflt = pd.DataFrame(gt.T)

        # dftest = pd.DataFrame(ltest.T)
        # for i in range(0, 4):
        #     data_to_img(dftest.iloc[:, i], 'rcc/testconv{}'.format(i))


        # dflt.insert(0, 'dif', full.iloc[:, 72 + 3])
        # dflt.insert(0, 'y', full.y)
        # dflt.insert(0, 'x', full.x)
        # figsave(dflt, dflt.columns[2], dflt.columns[2 + 3], v=0.4, p=5, name='j9rcc-76gt-73gt.png')
        # figsave(dflt, dflt.columns[2], dflt.columns[5 + 3], v=0.4, p=8, name='j9rcc-79gt-73gt.png')
        # figsave(dflt, dflt.columns[2], dflt.columns[8 + 3], v=0.4, p=11, name='j9rcc-82gt-73gt.png')
        # figsave(dflt, dflt.columns[2], dflt.columns[11 + 3], v=0.4, p=14, name='j9rcc-85gt-73gt.png')

        for i in range(0, 1):
            # pass
            # display(dflt, dflt.columns[i], 0, name='gan/funcgt.png')
            data_to_img(dflt.iloc[:, i], 'gan/7caddldgt1500{}'.format(step), grid_shape, grid_shape)

        dflp = pd.DataFrame(lprediction.T)
        # dflp_test = pd.DataFrame(lpreds_test)
        # dflp.insert(0, 'dif', full.iloc[:, 72 + 3])
        # dflp.insert(0, 'y', full.y)
        # dflp.insert(0, 'x', full.x)
        # figsave(dflp, dflp.columns[2], dflp.columns[2 + 3], v=0.4, p=5, name='j9rcc-76p-73gt.png')
        # figsave(dflp, dflp.columns[2], dflp.columns[5 + 3], v=0.4, p=8, name='j9rcc-79p-73gt.png')
        # figsave(dflp, dflp.columns[2], dflp.columns[8 + 3], v=0.4, p=11, name='j9rcc-82p-73gt.png')
        # figsave(dflp, dflp.columns[2], dflp.columns[11 + 3], v=0.4, p=14, name='j9rcc-85p-73gt.png')

        for i in range(0, 1):
            # pass
            # display(dflp, dflp.columns[i], 0, name='gan/funcp.png')
            data_to_img(dflp.iloc[:, i], 'gan/7caddldp1500{}'.format(step), grid_shape, grid_shape)
        step += 1
        break
print('done')
