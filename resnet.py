import numpy as np
import torch
import torch.nn as nn
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
from scatterFig import data_to_img

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def split_sequence(sequence, n_steps, jump):
    x1, x2 = list(), list()
    for i in range(0, len(sequence), 5):
        end_ix = i + n_steps * jump + 1
        if end_ix > len(sequence):
            break
        for j in range(n_steps):
            k1 = i + j * jump
            k2 = k1 + 1
            seq_x1, seq_x2 = sequence[k1], sequence[k2]
            x1.append(seq_x1)
            x2.append(seq_x2)
    return x1, x2


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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 这是残差网络中的basicblock，实现的功能如下方解释：
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes代表输入通道数，planes代表输出通道数。
        super(BasicBlock, self).__init__()
        # Conv1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # F(x)+x
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数的倍乘

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # conv1   1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # conv2   3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # conv3   1x1
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# resnet18
def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


# resnet34
def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


# resnet50
def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


# resnet101
def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


# resnet152
def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        # layers=参数列表 block选择不同的类
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 1.conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 3.conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 4.conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.g(x)


class Model(nn.Module):
    def __init__(self, encoder, pjh):
        super(Model, self).__init__()
        self._encoder = encoder
        self._pjh = pjh

    def forward(self, x):
        x = self._encoder(x)
        x = self._pjh(x)
        return x


def similarity(x1, x2):
    sim = torch.cosine_similarity(x1, x2, dim=0)
    sim = torch.exp(sim)
    return sim


def I(z1, z2, i, j):
    up = similarity(z1[i], z2[j])
    down = .0
    for m in range(len(z1)):
        if m != i:
            down += similarity(z1[i], z1[m])
    for n in range(len(z2)):
        down += similarity(z1[i], z2[n])
    return -torch.log(up / down)


def L(z1, z2):
    N = len(z1)
    loss = .0
    for i in range(N):
        loss += I(z1, z2, i, i) + I(z2, z1, i, i)
    loss /= 2 * N
    return loss


def Train():
    train_loss = .0
    for idx, (x1, x2) in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = x1.unsqueeze(1).cuda()
        x2 = x2.unsqueeze(1).cuda()
        z1 = model(x1)
        z2 = model(x2)
        loss = L(z1, z2)
        loss.backward()
        optimizer.step()
        train_loss += loss
    train_loss = train_loss / len(train_loader)
    print("train_loss:", train_loss.detach().cpu().numpy())


if __name__ == '__main__':
    df1 = pd.read_csv('uk190624.csv').fillna(0).iloc[:, 2:].T.values
    df2 = pd.read_csv('uk220101.csv').fillna(0).iloc[:, 2:].T.values
    df = np.concatenate((df1, df2), axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    data = torch.Tensor(df).T
    grid_shape = int(math.sqrt(data.shape[0]))
    timestep = data.shape[1]
    inp = data[:, 0].reshape(grid_shape, grid_shape).unsqueeze(0)
    for i in range(1, timestep):
        tmp = data[:, i].reshape(grid_shape, grid_shape).unsqueeze(0)
        inp = torch.cat((inp, tmp), 0)

    train_set = inp
    train_x1, train_x2 = split_sequence(train_set, n_steps=4, jump=50)
    # for i in range(len(train_x1)):
    #     data_to_img(train_x1[i], 'clgen/{}_x1'.format(i), grid_shape, grid_shape)
    #     data_to_img(train_x2[i], 'clgen/{}_x2'.format(i), grid_shape, grid_shape)

    train = MoveDataset(train_x1, train_x2)
    train_loader = DataLoader(train, batch_size=4, shuffle=False)

    encoder = resnet152().cuda()
    pjh = ProjectionHead().cuda()
    model = Model(encoder, pjh).cuda()
    epochs = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(epochs):
        print('epoch {}/{}'.format(i + 1, epochs))
        Train()
    torch.save(model, 'cl2.pt')
