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

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1f = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.conv1s = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2f = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.conv2s = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3f = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.conv3s = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4f = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.conv4s = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        self.pool4 = nn.MaxPool2d((2, 2))

        self.conv5f = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.conv5s = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        self.up6 = nn.Upsample(size=(32, 32))
        self.conv6 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.up7 = nn.Upsample(size=(64, 64))
        self.conv7 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.up8 = nn.Upsample(size=(128, 128))
        self.conv8 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.up9 = nn.Upsample(size=(256, 256))
        self.conv9 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.outputs = nn.Conv2d(2, 6, kernel_size=(1, 1))

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

    def forward(self, inputs):
        conv1f = self.conv1f(inputs)
        conv1s = self.conv1s(conv1f)
        pool1 = self.pool1(conv1s)

        conv2f = self.conv2f(pool1)
        conv2s = self.conv2s(conv2f)
        pool2 = self.pool2(conv2s)

        conv3f = self.conv3f(pool2)
        conv3s = self.conv3s(conv3f)
        pool3 = self.pool3(conv3s)

        conv4f = self.conv4f(pool3)
        conv4s = self.conv4s(conv4f)
        pool4 = self.pool4(conv4s)

        conv5f = self.conv5f(pool4)
        conv5s = self.conv5s(conv5f)

        up6 = torch.cat((self.up6(conv5s), conv4s), dim=1)
        conv6 = self.conv6(up6)

        up7 = torch.cat((self.up7(conv6), conv3s), dim=1)
        conv7 = self.conv7(up7)

        up8 = torch.cat((self.up8(conv7), conv2s), dim=1)
        conv8 = self.conv8(up8)

        up9 = torch.cat((self.up9(conv8), conv1s), dim=1)
        conv9 = self.conv9(up9)

        outputs = self.outputs(conv9)
        return outputs


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
        inputs = inputs.float().cuda()  # C2D + FC, C1D, LSTM
        tlabels = tlabels.float().cuda()  # C2D + C1D, LSTM
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
            inputs = inputs.float().cuda()  # C2D + FC, C1D, LSTM
            tlabels = tlabels.float().cuda()  # C2D + C1D, LSTM
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

    torch.save(model, 'rain.pt')
