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

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# device = torch.device("cuda:0")
n_steps = 4
multi = 6
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


class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(512, 512)
        self.tout = nn.Sequential(
            nn.Linear(n_steps, 256),
            nn.Linear(256, 16),
            nn.Linear(16, multi)
        )

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.tout(x.permute(2, 1, 0)).permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.sample1 = nn.PixelUnshuffle(2)
        self.unsample1 = nn.PixelShuffle(2)
        self.res1 = nn.Linear(4, 6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.sample2 = nn.PixelUnshuffle(2)
        self.unsample2 = nn.PixelShuffle(2)
        self.res2 = nn.Linear(4, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2)
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.sample3 = nn.PixelUnshuffle(2)
        self.unsample3 = nn.PixelShuffle(2)
        self.res3 = nn.Linear(4, 6)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2)
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.sample4 = nn.PixelUnshuffle(2)
        self.unsample4 = nn.PixelShuffle(2)
        self.grunet = GRUNet()
        self.loss = nn.Linear(2, 1)

    def low_level(self, x):
        steps = len(x)
        x = self.grunet(x.permute(1, 0, 2, 3).reshape(512, steps, -1).permute(1, 2, 0))

        return x

    def forward(self, inputs):
        hinputs1 = self.conv1(inputs)
        linputs1 = self.sample1(inputs)
        res1 = self.res1(torch.cat((linputs1, hinputs1), dim=1).permute(1, 2, 3, 0)).permute(3, 0, 1, 2)

        hinputs2 = self.conv2(hinputs1)
        linputs2 = self.sample1(linputs1)
        res2 = self.res2(torch.cat((linputs2, hinputs2), dim=1).permute(1, 2, 3, 0)).permute(3, 0, 1, 2)

        hinputs3 = self.conv3(hinputs2)
        linputs3 = self.sample1(linputs2)
        res3 = self.res3(torch.cat((linputs3, hinputs3), dim=1).permute(1, 2, 3, 0)).permute(3, 0, 1, 2)

        hinputs4 = self.conv4(hinputs3)
        linputs4 = self.sample1(linputs3)
        cat4 = torch.cat((linputs4, hinputs4), dim=1)

        lpreds = self.low_level(cat4).permute(1, 0, 2).reshape(multi, -1, 16, 16)

        lpreds4 = self.tconv4(lpreds)
        lpreds4 = self.unsample4(lpreds4)

        lpreds3 = self.tconv3(torch.cat((lpreds4, res3), dim=1))
        lpreds3 = self.unsample3(lpreds3)

        lpreds2 = self.tconv2(torch.cat((lpreds3, res2), dim=1))
        lpreds2 = self.unsample2(lpreds2)

        lpreds1 = self.tconv1(torch.cat((lpreds2, res1), dim=1))
        lpreds1 = self.unsample1(lpreds1)

        out = self.loss(lpreds1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

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
        lpreds = model(inputs)
        loss = logcosh(lpreds, tlabels)
        loss.backward()
        optimizer.step()

        # print_param(model.named_parameters())

        hrunning_loss += loss
    htrain_loss = hrunning_loss / len(train_loader)

    print(f'htrain_loss {htrain_loss}')


def Test():
    running_loss = .0

    model.eval()
    with torch.no_grad():
        for idx, (inputs, tlabels) in enumerate(test_loader):
            optimizer.zero_grad()
            inputs = inputs.float().permute(1, 0, 2, 3).cuda()  # C2D + FC, C1D, LSTM
            tlabels = tlabels.float().permute(1, 0, 2, 3).cuda()  # C2D + C1D, LSTM
            lpreds = model(inputs)
            lprediction = scaler.inverse_transform(lpreds.reshape(multi, -1).cpu())
            gt = scaler.inverse_transform(tlabels.reshape(multi, -1).cpu())
            loss = math.sqrt(mean_squared_error(lprediction, gt))
            running_loss += loss

        test_loss = running_loss / len(test_loader)
        print(f'valid_loss {test_loss}')


if __name__ == '__main__':
    # df = pd.read_csv('gussian_full.csv')
    # df = pd.read_csv('uk.csv')
    df = pd.read_csv('uk190624-256.csv').fillna(0).iloc[:, 2:].T.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    data = torch.Tensor(df).T
    # grid_shape = int(math.sqrt(data.shape[0]))
    timestep = data.shape[1]
    inp = data[:, 0].reshape(256, 256).unsqueeze(0)  # C2D + FC, C1D, LSTM
    # # inp = data[:, 0].unsqueeze(0)  # FC, Conv1D, LSTM
    for i in range(1, timestep):
        tmp = data[:, i].reshape(256, 256).unsqueeze(0)  # C2D + FC, C1D, LSTM
        # tmp = data[:, i].unsqueeze(0)  # FC, Conv1D, LSTM
        inp = torch.cat((inp, tmp), 0)

    slice = int(timestep * 0.8)
    # inp = inp[:, 96:224, 113:241]
    train_set = inp[:slice]
    test_set = inp[slice - (n_steps * jump + multi - 1):]

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

    torch.save(model, 'rccres.pt')
