import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
import os
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.utils import spectral_norm
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from resnet import similarity, Model, ResNet, Bottleneck, ProjectionHead

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4, 5, 6, 7'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--local_rank", type=int, default=0)  # 增加local_rank
parser.add_argument('--epoch_size', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--step_len', type=int, default=4)
parser.add_argument('--predict_len', type=int, default=6)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
epochs = args.epoch_size
batch_size = args.batch_size
n_steps = args.step_len
multi = args.predict_len
dist.init_process_group("nccl", init_method='env://')

def load_m(name):
    file_name = os.path.join(name)
    with open(file_name, 'rb') as f:
        model = torch.load(f, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu',
                                            'cuda:4': 'cpu', 'cuda:5': 'cpu', 'cuda:6': 'cpu', 'cuda:7': 'cpu'})
        # model = torch.load(f).module.cpu()
    return model


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


def logcosh(true, pred):
    loss = torch.log(torch.cosh(pred - true))
    return torch.sum(loss)


def loss_hinge_disc(score_generated, score_real):
    l1 = F.relu(1.0 - score_real)
    loss = torch.mean(l1)
    l2 = F.relu(1.0 + score_generated)
    loss += torch.mean(l2)
    return loss


def grid_cell_regularizer(generated_samples, batch_targets):
    gen_mean = torch.mean(generated_samples, dim=0)
    weights = torch.clip(batch_targets, 0.0, 24.0)
    loss = torch.mean(torch.abs(gen_mean - batch_targets) * weights)
    return loss


def loss_hinge_gen(score_generated):
    loss = -torch.mean(score_generated)
    return loss


def get_conv_layer(conv_type="standard"):
    if conv_type == "standard":
        conv_layer = nn.Conv2d
    elif conv_type == "3d":
        conv_layer = nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


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
        self.gru = nn.GRU(193, 512)
        # self.gru = nn.GRU(512, 512)
        self.dout = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 16),
            nn.Linear(16, 1)
        )
        self.tout = nn.Sequential(
            nn.Linear(n_steps, 256),
            nn.Linear(256, 16),
            nn.Linear(16, multi)
        )

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dout(x).permute(2, 1, 0)
        x = self.tout(x).permute(0, 2, 1)
        # x = self.tout(x.permute(2, 1, 0)).permute(0, 2, 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 11, padding=5)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.grunet = GRUNet()
        self.loss1 = nn.GRU(multi, 64)
        self.loss2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 4),
            nn.Linear(4, 1)
        )

    def low_level(self, x):
        steps = len(x)
        x = self.grunet(x.permute(1, 0, 2, 3).reshape(193, steps, -1).permute(1, 2, 0))
        return x

    def forward(self, inputs):
        hinputs1 = self.conv1(inputs)
        hinputs2 = self.conv2(hinputs1)
        linputs = torch.cat((hinputs2, hinputs1, inputs), 1)
        lpreds = self.low_level(linputs)
        lpreds, _ = self.loss1(lpreds.permute(0, 2, 1))
        lpreds = self.loss2(lpreds).permute(0, 2, 1)
        return lpreds


class Generator_rcc(nn.Module):
    def __init__(self):
        super(Generator_rcc, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Sigmoid()
        )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Sigmoid()
        )
        self.sample1 = nn.PixelUnshuffle(2)
        self.unsample1 = nn.PixelShuffle(2)
        self.res1 = nn.Sequential(
            nn.Linear(n_steps, multi),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Sigmoid()
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Sigmoid()
        )
        self.sample2 = nn.PixelUnshuffle(2)
        self.unsample2 = nn.PixelShuffle(2)
        self.res2 = nn.Sequential(
            nn.Linear(n_steps, multi),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.Sigmoid()
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Sigmoid()
        )
        self.sample3 = nn.PixelUnshuffle(2)
        self.unsample3 = nn.PixelShuffle(2)
        self.res3 = nn.Sequential(
            nn.Linear(n_steps, multi),
            nn.Sigmoid()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.Sigmoid()
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Sigmoid()
        )
        self.sample4 = nn.PixelUnshuffle(2)
        self.unsample4 = nn.PixelShuffle(2)
        self.grunet = GRUNet()
        self.loss = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

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

        # lpreds = self.low_level(cat4).permute(1, 0, 2).reshape(multi, -1, 25, 25)
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


class Discriminator_Line(nn.Module):
    def __init__(self):
        super(Discriminator_Line, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_steps + 1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.model(x.permute(1, 2, 0))
        x = torch.mean(x, dim=1)
        return x


class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.line = nn.Sequential(
            nn.Linear(128, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.low = nn.Sequential(
            nn.Linear(n_steps + 1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.reshape(n_steps + 1, 1, 256, 256)
        x = self.conv1(x)
        x = self.conv2(x).reshape(n_steps + 1, 128, -1).permute(0, 2, 1)
        x = self.line(x).permute(1, 2, 0)
        loss = self.low(x)
        return loss


class Discriminator(nn.Module):
    def __init__(self, input_channels, num_spatial_frames=8, conv_type="standard"):
        super(Discriminator, self).__init__()
        self.spatial_discriminator = SpatialDiscriminator(
            input_channels=input_channels, num_timesteps=num_spatial_frames, conv_type=conv_type
        )
        self.temporal_discriminator = TemporalDiscriminator(
            input_channels=input_channels, conv_type=conv_type
        )

    def forward(self, inputs):
        spatial_loss = self.spatial_discriminator(inputs)
        temporal_loss = self.temporal_discriminator(inputs)
        return torch.cat([spatial_loss, temporal_loss], dim=1)


class TemporalDiscriminator(nn.Module):
    def __init__(self, input_channels, num_layers=3, conv_type="standard"):
        super(TemporalDiscriminator, self).__init__()
        self.downsample = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 48
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=internal_chn * input_channels,
            conv_type="3d",
            first_relu=False,
        )
        self.d2 = DBlock(
            input_channels=internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            conv_type="3d",
        )
        self.intermediate_dblocks = nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d_last = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )
        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.space2depth(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.d1(x)
        x = self.d2(x)
        x = x.permute(0, 2, 1, 3, 4)
        representations = []
        for idx in range(x.size(1)):
            rep = x[:, idx, :, :, :]
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d_last(rep)
            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            representations.append(rep)
        x = torch.stack(representations, dim=1)
        x = torch.sum(x, keepdim=True, dim=1)
        return x


class SpatialDiscriminator(nn.Module):
    def __init__(self, input_channels, num_timesteps, num_layers=3, conv_type="standard"):
        super(SpatialDiscriminator, self).__init__()
        self.num_timesteps = num_timesteps
        self.mean_pool = nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=2 * internal_chn * input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )
        self.fc = spectral_norm(nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x):
        idxs = torch.randint(low=0, high=x.size()[1], size=(self.num_timesteps,))
        representations = []
        for idx in idxs:
            rep = self.mean_pool(x[:, idx, :, :, :])
            rep = self.space2depth(rep)
            rep = self.d1(rep)
            for d in self.intermediate_dblocks:
                tmp = d(rep)
                rep = tmp
            rep = self.d6(rep)
            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            representations.append(rep)
        x = torch.stack(representations, dim=1)
        x = torch.sum(x, keepdim=True, dim=1)
        return x


class DBlock(nn.Module):
    def __init__(self, input_channels, output_channels, conv_type="standard", first_relu=True, keep_same_output=False):
        super(DBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        if conv_type == "3d":
            self.pooling = nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
        )
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x
        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x
        return x



def Train():
    d_loss = .0
    g_loss = .0
    grid_loss = .0
    generator.train()
    discriminator.train()
    for idx, (images, future_images) in enumerate(train_loader):
        images = images.float().permute(1, 0, 2, 3).cuda()
        future_images = future_images.float().permute(1, 0, 2, 3).cuda()
        # inputs = images.unsqueeze(0)
        inputs = images.reshape(n_steps, 1, -1)
        # labels = future_images[-1].reshape(1, 1, 1, 256, 256)
        labels = future_images[-1].reshape(1, 1, -1)
        for _ in range(2):
            # predictions = generator(images).reshape(1, 1, 1, 256, 256)
            predictions = generator(images).reshape(1, 1, -1)
            # generated_sequence = torch.cat([inputs, predictions], dim=1)
            generated_sequence = torch.cat([inputs, predictions], dim=0)
            # real_sequence = torch.cat([inputs, labels], dim=1)
            real_sequence = torch.cat([inputs, labels], dim=0)
            # concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)
            # concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=1)
            # concatenated_outputs = discriminator(concatenated_inputs)
            # score_real, score_generated = torch.split(concatenated_outputs, 1, dim=0)
            score_real = discriminator(real_sequence)
            score_generated = discriminator(generated_sequence)
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            # discriminator_loss = -(torch.mean(score_real) - torch.mean(score_generated))
            d_opt.zero_grad()
            discriminator_loss.backward()
            d_opt.step()
        # predictions = [generator(images).reshape(1, 1, 1, 256, 256) for _ in range(1)]
        predictions = [generator(images).reshape(1, 1, -1) for _ in range(2)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), labels)
        # generated_sequence = [torch.cat([inputs, x], dim=1) for x in predictions]
        generated_sequence = [torch.cat([inputs, x], dim=0) for x in predictions]
        # real_sequence = torch.cat([inputs, labels], dim=1)
        real_sequence = torch.cat([inputs, labels], dim=0)
        generated_scores = []
        for g_seq in generated_sequence:
            # concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            # concatenated_inputs = torch.cat([real_sequence, g_seq], dim=1)
            # concatenated_outputs = discriminator(concatenated_inputs)
            # score_real, score_generated = torch.split(concatenated_outputs, 1, dim=0)
            score_generated = discriminator(g_seq)
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + 20 * grid_cell_reg
        g_opt.zero_grad()
        generator_loss.backward()
        g_opt.step()
        d_loss += discriminator_loss
        g_loss += generator_loss
        grid_loss += grid_cell_reg
    d_loss = d_loss / len(train_loader)
    g_loss = g_loss / len(train_loader)
    grid_loss = grid_loss / len(train_loader)
    print(f'train/d_loss:{d_loss}, train/g_loss:{g_loss}, train/grid_loss:{grid_loss}')


def Test():
    test_loss = .0
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for idx, (inputs, tlabels) in enumerate(train_loader):
            inputs = inputs.float().permute(1, 0, 2, 3).cuda()
            tlabels = tlabels.float().permute(1, 0, 2, 3).cuda()
            lpreds = generator(inputs)
            lprediction = scaler.inverse_transform(lpreds.reshape(1, -1).cpu())
            gt = scaler.inverse_transform(tlabels[-1].reshape(1, -1).cpu())
            loss = math.sqrt(mean_squared_error(lprediction, gt))
            test_loss += loss
        test_loss = test_loss / len(train_loader)
        print(f'test/loss:{test_loss}')


if __name__ == '__main__':
    df180101 = pd.read_csv('uk180101.csv').fillna(0).iloc[:, 2:].T.values
    df180102 = pd.read_csv('uk180102.csv').fillna(0).iloc[:, 2:].T.values
    df190628 = pd.read_csv('uk190628.csv').fillna(0).iloc[:, 2:].T.values
    df190629 = pd.read_csv('uk190629.csv').fillna(0).iloc[:, 2:].T.values
    df190630 = pd.read_csv('uk190630.csv').fillna(0).iloc[:, 2:].T.values
    df220101 = pd.read_csv('uk220101.csv').fillna(0).iloc[:, 2:].T.values
    df190624 = pd.read_csv('uk190624.csv').fillna(0).iloc[:, 2:].T.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(
        np.concatenate((df180101, df180102, df190628, df190629, df190630, df220101, df190624), axis=0))
    data = torch.Tensor(df).T
    grid_shape = int(math.sqrt(data.shape[0]))
    timestep = data.shape[1]
    inp = data[:, 0].reshape(grid_shape, grid_shape).unsqueeze(0)
    for i in range(1, timestep):
        tmp = data[:, i].reshape(grid_shape, grid_shape).unsqueeze(0)
        inp = torch.cat((inp, tmp), 0)

    slice = 288 * 6 + 220
    label_norain1 = inp[287]
    label_hrain = inp[288 * 6 + 52]
    label_lrain = inp[288 * 6]
    train_set = inp[:slice]
    test_set = inp[slice - 4:]

    model = load_m('cl.pt')
    model.eval()
    with torch.no_grad():
        norain1 = label_norain1.reshape(1, 1, 256, 256)
        hrain = label_hrain.reshape(1, 1, 256, 256)
        lrain = label_lrain.reshape(1, 1, 256, 256)
        znr1 = model(norain1)[0]
        zhr = model(hrain)[0]
        zlr = model(lrain)[0]
        res = []
        final = []
        for i in range(len(train_set)):
            image = train_set[i].reshape(1, 1, 256, 256)
            zim = model(image)[0]
            sim_nr1 = similarity(zim, znr1)
            sim_hr = similarity(zim, zhr)
            sim_lr = similarity(zim, zlr)
            if sim_lr > sim_nr1 or sim_hr > sim_nr1:
                res.append(image.squeeze(1))
            elif res is not None:
                if len(res) >= 10:
                    final.append(res)
                res = []

    for i in range(len(final)):
        if i == 0:
            train_x, train_y = split_sequence(torch.cat(final[i], dim=0), 4, 6)
        else:
            tmp_x, tmp_y = split_sequence(torch.cat(final[i], dim=0), 4, 6)
            train_x += tmp_x
            train_y += tmp_y

    test_x, test_y = split_sequence(test_set, n_steps=n_steps, multi=multi)

    train = MoveDataset(train_x, train_y)
    test = MoveDataset(test_x, test_y)

    train_sampler = DistributedSampler(train)
    test_sampler = DistributedSampler(test)

    train_loader = DataLoader(train, batch_size=1, sampler=train_sampler, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=1, sampler=test_sampler, shuffle=False, num_workers=8, pin_memory=True)

    # model = Model().cuda()
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    generator = Generator().cuda()
    discriminator = Discriminator_Line().cuda()
    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank], broadcast_buffers=False,
                                                    find_unused_parameters=True)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank],
                                                        broadcast_buffers=False, find_unused_parameters=True)
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.0, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=g_opt, gamma=0.98)

    for i in range(epochs):
        print('epoch {}/{}'.format(i + 1, epochs))
        train_sampler.set_epoch(i)
        test_sampler.set_epoch(i)
        Train()
        Test()
        gc.collect()
        # my_lr_scheduler.step()
        if (i + 1) % 100 == 0:
            if args.local_rank == 0:
                torch.save(generator, 'v4clcaddld{}.pt'.format(i))
