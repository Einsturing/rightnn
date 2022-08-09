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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
n_steps = 4
multi = 1


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
        self.lstm = nn.LSTM(3, 64, 2)
        self.out = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 16),
            nn.Linear(16, 1)
        )
        self.dout = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 16),
            nn.Linear(16, 3)
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
        self.conv1 = nn.Conv2d(1, 1, 11, padding=5)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        # self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        # self.conv4 = nn.Conv2d(1, 1, 3, padding=1)
        # self.dconv = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, 8),
        # )
        # self.hfcnet = FCNet()
        # self.lfcnet = FCNet()
        # self.hc1dnet = C1DNet()
        # self.lc1dnet = C1DNet()
        self.lstmnet = LSTMNet()

    # def FC(self, x):
    #     return self.fcnet(x).permute(3, 0, 1, 2)

    # def HC1D(self, x):
    #     return self.hc1dnet(x.squeeze(0).permute(0, 2, 1))

    # def high_level(self, x):
    #     # x = self.conv(x).permute(1, 2, 3, 0)  # C2D + FC
    #     # x = self.conv(x).permute(1, 0, 2, 3).reshape(1, 16, -1)  # C2D + C1D
    #     x = self.conv(x).permute(1, 0, 2, 3).reshape(1, 16, -1).permute(0, 2, 1)  # C2D + LSTM
    #     # x = self.hfcnet(x).permute(3, 0, 1, 2)  # C2D + FC
    #     # x = self.hc1dnet(x)  # C2D + C1D
    #     x = self.lstmnet(x)  # C2D + LSTM
    #     return x

    def low_level(self, x):
        # x = self.lfcnet(x.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)  # C2D + FC
        # x = self.lc1dnet(x.permute(1, 0, 2, 3).reshape(1, 16, -1))  # C2D + C1D
        # x = self.lstmnet(x.permute(1, 0, 2, 3).reshape(2, n_steps, -1).permute(0, 2, 1))  # C2D + LSTM
        # x = self.lstmnet(x.permute(1, 0, 2, 3).reshape(3, n_steps, -1).permute(0, 2, 1))  # C2D + LSTM
        x = self.lstmnet(x.permute(1, 0, 2, 3).reshape(3, n_steps, -1).permute(1, 2, 0))
        # x = self.lstmnet(x.permute(1, 0, 2, 3).reshape(5, n_steps, -1).permute(1, 2, 0))
        # x = self.lstmnet(x.permute(1, 0, 2, 3).reshape(4, n_steps, -1).permute(0, 2, 1))  # C2D + LSTM

        return x

    def multi_preds(self, x, steps):
        tmp_window = x
        # tmp_list = []
        for i in range(steps):
            next_step = self.low_level(tmp_window)
            # tmp_list.append(next_step)
            tmp_window = torch.cat((tmp_window[torch.arange(tmp_window.size(0)) != 0, :, :, :],
                                    next_step.permute(1, 0, 2).reshape(1, 3, 320, 355)), 0)
        # return torch.cat(tmp_list, 1)
        return next_step


def print_param(params):
    for name, parameter in params:
        if not parameter.requires_grad:
            continue
        print(name)
        print(parameter)


def get_n_nearst(data, n):
    nearst_n = []
    for source in data:
        source_id = source[0]
        source_x = source[1]
        source_y = source[2]
        node_dict = {}
        for node in data:
            node_id = int(node[0].item())
            if node_id == source_id:
                continue
            node_x = node[1]
            node_y = node[2]
            dist = get_eu_dist(source_x, node_x, source_y, node_y)
            node_dict[node_id] = dist
        sorted_dict = sorted(node_dict.items(), key=lambda x: x[1])
        nearst_n.append(sorted_dict[:n])
    return nearst_n


def split_sequence(sequence, n_steps, multi=1):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix + multi > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix + multi]
        x.append(seq_x)
        y.append(seq_y)
    return x, y


def get_eu_dist(x1, x2, y1, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_n_nearst(data, n):
    nearst_n = []
    for source in data:
        source_id = source[0]
        source_x = source[1]
        source_y = source[2]
        node_dict = {}
        for node in data:
            node_id = int(node[0].item())
            if node_id == source_id:
                continue
            node_x = node[1]
            node_y = node[2]
            dist = get_eu_dist(source_x, node_x, source_y, node_y)
            node_dict[node_id] = dist
        sorted_dict = sorted(node_dict.items(), key=lambda x: x[1])
        nearst_n.append(sorted_dict[:n])
    return nearst_n


def get_all_conv(tmp, conv):
    all_conv = tmp[:, conv[0]].unsqueeze(1)
    for i in range(1, 1024):
        i_conv = conv[i]
        tmp2 = tmp[:, i_conv].unsqueeze(1)
        all_conv = torch.cat((all_conv, tmp2), dim=1)
    return all_conv


def Train():
    hrunning_loss = .0
    model.train()

    for idx, (inputs, tlabels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.float().permute(1, 0, 2, 3).to(device)  # C2D + FC, C1D, LSTM
        hinputs = model.conv1(inputs)  # C2D + C1D, LSTM
        hinputs2 = model.conv2(hinputs)
        # hinputs3 = model.conv3(hinputs2)
        tlabels = tlabels.float().permute(1, 0, 2, 3).to(device)  # C2D + C1D, LSTM
        hlabels = model.conv1(tlabels)  # C2D + FC
        hlabels2 = model.conv2(hlabels)
        # hlabels3 = model.conv3(hlabels2)
        # linputs = torch.cat((hinputs, inputs), 1)
        # llabels = torch.cat((hlabels, tlabels), 1)
        linputs = torch.cat((hinputs2, hinputs, inputs), 1)
        llabels = torch.cat((hlabels2, hlabels, tlabels), 1)
        # linputs = torch.cat((hinputs, hinputs2, hinputs3, inputs), 1)
        # llabels = torch.cat((hlabels, hlabels2, hlabels3, tlabels), 1)
        lpreds = model.low_level(linputs)
        # loss = criterion(lpreds, llabels.reshape(1, 2, 1024).permute(1, 0, 2))
        loss = criterion(lpreds[2], tlabels.reshape(multi, -1))
        # loss = criterion(lpreds, llabels.reshape(1, 4, 1024).permute(1, 0, 2))
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
            inputs = inputs.float().permute(1, 0, 2, 3).to(device)  # C2D + FC, C1D, LSTM
            hinputs = model.conv1(inputs)
            hinputs2 = model.conv2(hinputs)
            # hinputs3 = model.conv3(hinputs2)
            tlabels = tlabels.float().permute(1, 0, 2, 3).to(device)  # C2D + C1D, LSTM
            hlabels = model.conv1(tlabels)  # C2D + FC
            hlabels2 = model.conv2(hlabels)
            # hlabels3 = model.conv3(hlabels2)
            # linputs = torch.cat((hinputs, inputs), 1)
            # llabels = torch.cat((hlabels, tlabels), 1)
            linputs = torch.cat((hinputs2, hinputs, inputs), 1)
            llabels = torch.cat((hlabels2, hlabels, tlabels), 1)
            # linputs = torch.cat((hinputs, hinputs2, hinputs3, inputs), 1)
            # llabels = torch.cat((hlabels, hlabels2, hlabels3, tlabels), 1)
            lpreds = model.low_level(linputs)
            lprediction = scaler.inverse_transform(lpreds[2].cpu())
            gt = scaler.inverse_transform(llabels[:, 2, :, :].reshape(multi, -1).cpu())
            # lprediction = lpreds[2]
            # gt = llabels[:, 2, :, :].reshape(multi, -1)
            # loss = criterion(lpreds, llabels.reshape(1, 2, 1024).permute(1, 0, 2))
            loss = math.sqrt(mean_squared_error(lprediction, gt))
            # loss = criterion(lprediction, gt)
            # loss = criterion(lpreds, llabels.reshape(1, 4, 1024).permute(1, 0, 2))
            running_loss += loss

        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)
        print(f'valid_loss {test_loss}')


if __name__ == '__main__':
    # df = pd.read_csv('gussian_full.csv')
    df = pd.read_csv('radar_full.csv').fillna(0).iloc[:, 3:].T.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    data = torch.Tensor(df).T
    grid_shape = int(math.sqrt(data.shape[0]))
    timestep = data.shape[1]
    inp = data[:, 0].reshape(320, 355).unsqueeze(0)  # C2D + FC, C1D, LSTM
    # # inp = data[:, 0].unsqueeze(0)  # FC, Conv1D, LSTM
    for i in range(1, timestep):
        tmp = data[:, i].reshape(320, 355).unsqueeze(0)  # C2D + FC, C1D, LSTM
        # tmp = data[:, i].unsqueeze(0)  # FC, Conv1D, LSTM
        inp = torch.cat((inp, tmp), 0)

    # inp = inp[:, 96:224, 113:241]
    train_set = inp[:39]
    test_set = inp[39 - (n_steps + multi - 1):]

    train_x, train_y = split_sequence(train_set, n_steps=n_steps, multi=multi)
    test_x, test_y = split_sequence(test_set, n_steps=n_steps, multi=multi)

    train = MoveDataset(train_x, train_y)
    test = MoveDataset(test_x, test_y)

    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    model = Model().to(device)
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

    torch.save(model, 'radarlj6.pt')
