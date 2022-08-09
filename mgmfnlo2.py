import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
import os
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# device = torch.device("cuda:0")
n_steps = 2
multi = 2
jump = 1


class MoveDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item, label


class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(193, 64)
        self.out = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 16),
            nn.Linear(16, 1)
        )
        self.dout = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 16),
            nn.Linear(16, 1)
        )
        self.tout = nn.Sequential(
            nn.Linear(n_steps, 256),
            nn.Linear(256, 16),
            nn.Linear(16, multi)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = self.out(x).permute(0, 2, 1)
        x = self.dout(x).permute(2, 1, 0)
        x = self.tout(x).permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 11, padding=5)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        # self.conv4 = nn.Conv2d(1, 1, 3, padding=1)
        self.lstmnet = LSTMNet()
        self.loss1 = nn.GRU(multi, 64)
        self.loss2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 4),
            nn.Linear(4, 1)
        )

    def low_level(self, x):
        steps = len(x)
        x = self.lstmnet(x.permute(1, 0, 2, 3).reshape(193, steps, -1).permute(1, 2, 0))

        return x

    def forward(self, inputs, tlabels):
        hinputs = self.conv1(inputs)
        hinputs2 = self.conv2(hinputs)
        hlabels = self.conv1(tlabels)
        hlabels2 = self.conv2(hlabels)
        linputs = torch.cat((hinputs2, hinputs, inputs), 1)
        llabels = torch.cat((hlabels2, hlabels, tlabels), 1)
        lpreds = self.low_level(linputs)
        lpreds, _ = self.loss1(lpreds.permute(0, 2, 1))
        lpreds = self.loss2(lpreds).permute(0, 2, 1)
        return lpreds

    def multi_preds(self, x, steps):
        tmp_window = x
        tmp_list = []
        for i in range(steps):
            next_step = self.low_level(tmp_window)
            # tmp_list.append(next_step)
            tmp_window = torch.cat((tmp_window[torch.arange(tmp_window.size(0)) != 0, :, :, :],
                                    next_step.permute(1, 0, 2).reshape(1, 3, 320, 355)), 0)
        # return torch.cat(tmp_list, 1)
        return next_step


def logcosh(true, pred):
    loss = torch.log(torch.cosh(pred - true))
    return torch.sum(loss)


def print_param(params):
    for name, parameter in params:
        if not parameter.requires_grad:
            continue
        print(name)
        print(parameter)


def split_sequence(sequence, n_steps, multi=1):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps * jump
        if end_ix + multi > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix:jump], sequence[end_ix:end_ix + multi]
        x.append(seq_x)
        y.append(seq_y)
    return x, y


def Train():
    hrunning_loss = .0
    model.train()

    for idx, (inputs, tlabels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.float().permute(1, 0, 2, 3).cuda()  # C2D + FC, C1D, LSTM
        tlabels = tlabels.float().permute(1, 0, 2, 3).cuda()  # C2D + C1D, LSTM
        lpreds = model(inputs, tlabels)
        # loss = criterion(lpreds.reshape(1, -1), tlabels.reshape(multi, -1)[-1])
        loss = logcosh(lpreds.reshape(1, -1), tlabels.reshape(multi, -1)[-1])
        loss.backward()
        optimizer.step()

        # print_param(model.named_parameters())

        hrunning_loss += loss
    htrain_loss = hrunning_loss / len(train_loader)
    htrain_losses.append(htrain_loss.detach().cpu().numpy())

    print(f'htrain_loss {htrain_loss}')


def Test():
    running_loss = .0

    model.eval()
    with torch.no_grad():
        for idx, (inputs, tlabels) in enumerate(test_loader):
            optimizer.zero_grad()
            inputs = inputs.float().permute(1, 0, 2, 3).cuda()  # C2D + FC, C1D, LSTM
            tlabels = tlabels.float().permute(1, 0, 2, 3).cuda()  # C2D + C1D, LSTM
            lpreds = model(inputs, tlabels)
            lprediction = scaler.inverse_transform(lpreds.reshape(1, -1).cpu())
            gt = scaler.inverse_transform(tlabels.reshape(multi, -1).cpu())
            loss = math.sqrt(mean_squared_error(lprediction[0], gt[-1]))
            running_loss += loss

        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)
        print(f'valid_loss {test_loss}')


if __name__ == '__main__':
    # df = pd.read_csv('gussian_full.csv')
    # df = pd.read_csv('uk.csv')
    df = pd.read_csv('j9_grid_true.csv').fillna(0).iloc[:, 3:].T.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    data = torch.Tensor(df).T
    grid_shape = int(math.sqrt(data.shape[0]))
    timestep = data.shape[1]
    inp = data[:, 0].reshape(grid_shape, grid_shape).unsqueeze(0)  # C2D + FC, C1D, LSTM
    # # inp = data[:, 0].unsqueeze(0)  # FC, Conv1D, LSTM
    for i in range(1, timestep):
        tmp = data[:, i].reshape(grid_shape, grid_shape).unsqueeze(0)  # C2D + FC, C1D, LSTM
        # tmp = data[:, i].unsqueeze(0)  # FC, Conv1D, LSTM
        inp = torch.cat((inp, tmp), 0)

    # slice = int(timestep * 0.8)
    # inp = inp[:, 96:224, 113:241]
    train_set = inp[:10]
    test_set = inp[10 - (n_steps * jump + multi - 1):]

    train_x, train_y = split_sequence(train_set, n_steps=n_steps, multi=multi)
    test_x, test_y = split_sequence(test_set, n_steps=n_steps, multi=multi)

    train = MoveDataset(train_x, train_y)
    test = MoveDataset(test_x, test_y)

    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    model = Model().cuda()
    # model = nn.DataParallel(model, device_ids=[0])
    # model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    criterion = nn.MSELoss()

    htrain_losses = []
    ltrain_losses = []
    test_losses = []
    epochs = 5000
    for i in range(epochs):
        print('epoch {}/{}'.format(i + 1, epochs))
        Train()
        Test()
        # if i + 1 % 1000 == 0:
        #     my_lr_scheduler.step()
        gc.collect()

    torch.save(model, 'j9true.pt')
