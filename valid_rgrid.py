import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
from scatterFig import figsave, display, data_to_img
import os
# from mgsf import Model, LSTMNet, split_sequence, MoveDataset
from gan import Model, GRUNet, Generator_rcc, Discriminator, split_sequence, MoveDataset, Discriminator_Line
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pyproj import Proj, transform

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
gridhw = 400

def load_m():
    file_name = os.path.join('usganrccld.pt')
    with open(file_name, 'rb') as f:
        model = torch.load(f, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu', 'cuda:4': 'cpu', 'cuda:5': 'cpu', 'cuda:6': 'cpu', 'cuda:7': 'cpu'})

    return model

n_steps = 4
multi = 6
jump = 1


# _TEMP_FOLDER = './xinjiu_all'
# data_file_list = os.listdir(_TEMP_FOLDER)
# data_file_list.sort(key=lambda x: int(x[:-12]))
# k = 0
# full = None
# for d in data_file_list:
#     df = pd.read_csv(os.path.join(_TEMP_FOLDER, d), header=None)
#     if not k:
#         df = df.iloc[:, :3]
#         full = df
#     else:
#         full.insert(full.shape[1], k + 2, df.iloc[:, 2])
#     k += 1
# df = full
df = pd.read_csv('uscovid.csv')
full = df
id = pd.DataFrame(torch.Tensor([i for i in range(len(df))]).numpy().astype(int))
# df.insert(0, 'id', id)
inProj = Proj(init='epsg:3857')
outProj = Proj(init='epsg:4326')
df.iloc[:, 1], df.iloc[:, 2] = transform(inProj, outProj, df.iloc[:, 1], df.iloc[:, 2])
source = df.iloc[:, 3:].to_numpy().T
df.rename(columns={0: 'x', 1: 'y'}, inplace=True)
# full.rename(columns={0: 'x', 1: 'y'}, inplace=True)
minX = df.x.min()
maxX = df.x.max()
skipX = (maxX - minX) / gridhw
minY = df.y.min()
maxY = df.y.max()
skipY = (maxY - minY) / gridhw
gridX = [minX]
gridY = [minY]
nextX, nextY = minX, minY
for i in range(gridhw):
    nextX = nextX + skipX
    nextY = nextY + skipY
    gridX.append(nextX)
    gridY.append(nextY)
    dfpoint = df.iloc[:, :3]

# X = pd.DataFrame(torch.Tensor([[gridX[j] for j in range(gridhw)] for i in range(gridhw)]).view(gridhw ** 2).numpy())
# Y = pd.DataFrame(torch.Tensor([[gridY[i] for j in range(gridhw)] for i in range(gridhw)]).view(gridhw ** 2).numpy())
#
# Y.insert(0, 'X', X)
# Y.iloc[:, 0], Y.iloc[:, 1] = transform(outProj, inProj, Y.iloc[:, 0], Y.iloc[:, 1])

mp = {}
for i in range(len(df)):
    tmpid = df.id[i]
    tmpX = df.x[i]
    tmpY = df.y[i]
    for locX in range(len(gridX) - 1):
        if gridX[locX] <= tmpX < gridX[locX + 1]:
            thisX = locX
            break
    for locY in range(len(gridY) - 1):
        if gridY[locY] <= tmpY < gridY[locY + 1]:
            thisY = locY
            break
    location = (locX, locY)
    mp[tmpid] = location


df = pd.read_csv('uscovid_grid_full.csv').fillna(0).iloc[:, 3:].T.values
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
data = torch.Tensor(df).T
grid_shape = int(math.sqrt(data.shape[0]))
timestep = data.shape[1]
# draw = pd.read_csv('once12-gt-pred-6071.csv')
# draw.insert(0, 'y', full.y)
# draw.insert(0, 'x', full.x)
# figsave(draw, draw.columns[1 + 2], draw.columns[25 + 2], v=0.4, name='once12-71p-59gt.png')
# figsave(draw, draw.columns[1 + 2], draw.columns[13 + 2], v=0.4, name='once12-71gt-59gt.png')
# figsave(draw, draw.columns[6], draw.columns[25 + 2], v=0.4, name='once12-71p-71gt.png')

# move = pd.read_csv('pst_full.csv')
# df = pd.read_csv('pst.csv').to_numpy()
# data = torch.Tensor(df)
inp = data[:, 0].reshape(grid_shape, grid_shape).unsqueeze(0)
for i in range(1, timestep):
    tmp = data[:, i].reshape(grid_shape, grid_shape).unsqueeze(0)  # C2D + FC, C1D, LSTM
    #     # tmp = data[:, i].unsqueeze(0)  # FC, Conv1D, LSTM
    inp = torch.cat((inp, tmp), 0)

slice = int(timestep * 0.8)
test_set = inp[slice - n_steps:]
test_x, test_y = split_sequence(test_set, n_steps=n_steps, multi=multi)
test = MoveDataset(test_x, test_y)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
model = load_m()

# groundtruth59 = inp[59].numpy().reshape(1, -1)
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
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param)

    step = 1
    for idx, (inputs, tlabels) in enumerate(test_loader):
        inputs = inputs.float().permute(1, 0, 2, 3)
        tlabels = tlabels.float().permute(1, 0, 2, 3)
        lpreds = model.predict(inputs)
        lprediction = scaler.inverse_transform(lpreds.reshape(multi, -1).cpu())
        gt = scaler.inverse_transform(tlabels.reshape(multi, -1).cpu())

        # pred = pd.DataFrame(lprediction)
        # pred.insert(0, 'gt', gt6071)
        # pred.to_csv('once12_pred.csv')

        # lp = lpreds[2].numpy().reshape(-1, 1)
        # gt = tlabels.numpy().reshape(-1, 1)
        rp, rgt = [], []
        for value in mp:
            tmpx = mp[value][0]
            tmpy = mp[value][1]
            tmploc = tmpx * 400 + tmpy
            rp.append(lprediction[-1, tmploc])
            rgt.append(gt[-1, tmploc])

        rmse = math.sqrt(mean_squared_error(rp, rgt))

        dflp = pd.DataFrame(np.array(rp))
        # dflp.insert(0, 'dif', full.iloc[:, 72 + 3])
        dflp.insert(0, 'y', full.y)
        dflp.insert(0, 'x', full.x)
        # figsave(dflp, dflp.columns[2], dflp.columns[3], v=0.4, name='t86p-73gt.png')
        for i in range(2, 3):
            # pass
            display(dflp, dflp.columns[i], 0, name='ganus p-rmse{}.png'.format(rmse))
            # data_to_img(dflt.iloc[:, i], 'gan/190629gt{}'.format(step), grid_shape, grid_shape)

        #
        # gt = scaler.inverse_transform(tlabels.reshape(1, -1))


        # rmse = math.sqrt(mean_squared_error(lprediction, gt))
        # gt = scaler.inverse_transform(llabels[:, 2, :, :].reshape(1, -1)).T
        # tmp_line = tmp.unsqueeze(1)[:, :, 0, :]
        # htmp = model.C2D(tmp.unsqueeze(1))
        # dfht = pd.DataFrame(htmp.numpy().reshape(625, -1))
        # dflt_line = pd.DataFrame(tmp_line.numpy().reshape(128, -1))
        dflt = pd.DataFrame(np.array(rgt))
        # dflt.insert(0, 'dif', full.iloc[:, 72 + 3])
        dflt.insert(0, 'y', full.y)
        dflt.insert(0, 'x', full.x)
        # figsave(dflt, dflt.columns[2], dflt.columns[3], v=0.4, name='t86gt-73gt.png')
        for i in range(2, 3):
            # pass
            display(dflt, dflt.columns[i], 0, name='ganus gt.png')
            # data_to_img(dflt.iloc[:, i], 'gan/190629gt{}'.format(step), grid_shape, grid_shape)

        # dflp.to_csv('lp.csv')
        # dflt.to_csv('gt.csv')
        # display(dfg2, dfg2.columns[3], 0, name='slxgrid-2layer-7+3k-5000epoch-p.png')
        # display(dfg2, dfg2.columns[2], 0, name='slxgrid-2layer-7+3k-5000epoch-gt.png')
        # figsave(dfg2, dfg2.columns[4], dfg2.columns[3], v=0.4, name='63p-60gt.png')
        # figsave(dfg2, dfg2.columns[4], dfg2.columns[2], v=0.4, name='63gt-60gt.png')
        # figsave(dfg2, dfg2.columns[2], dfg2.columns[3], v=0.4, name='63p-63gt.png')
        # dfg2.to_csv('63p63gt60gt113.csv')
print('done')
