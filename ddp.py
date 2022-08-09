import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import torch
import torch.nn as nn
import os.path
from torch.utils.data import DataLoader, Dataset
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--local_rank", type=int)  # 增加local_rank
parser.add_argument('--epoch_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=360)
parser.add_argument('--step_len', type=int, default=24)
parser.add_argument('--predict_len', type=int, default=13)
parser.add_argument('--kernel_size', type=int, default=15)
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--attribute_size', type=int, default=3)
parser.add_argument('--low_factor', type=float, default=1.0)
parser.add_argument('--high_factor', type=float, default=1.0)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
epoch_size = args.epoch_size
batch_size = args.batch_size
step_len = args.step_len
predict_len = args.predict_len
prefix = args.prefix
attribute_size = args.attribute_size
kernel_size = args.kernel_size
low_factor = args.low_factor
high_factor = args.high_factor
# In[94]:
batch_size = 6150
months = 73 + predict_len
epoch_step = 10
epoch_size = 10
attribute_size = 6
month_size = 30744
hidden_layer_size = 100
ignore_len = predict_len - 1
dist.init_process_group("nccl", init_method='env://')
device = args.local_rank
DEVICE = torch.device('cuda:'+str(device) if torch.cuda.is_available() else 'cpu')
val_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num = 8
batch_size = month_size // num
# DIR_ROOT = "/Users/stardust/data/SandWater-DATA/disease/disease/"



class SandWaterDataSet(Dataset):

    def __init__(self, start, end):
        self.DIR_ROOT = "/home/pclab/dlf/xinjiu_all/"
        print("epoch=%s,batch_size=%s,step_len=%s" % (epoch_size, batch_size, step_len))
        data_bed = []
        for f in sorted(glob.glob(self.DIR_ROOT + "*_20*-*.csv"),
                        key=lambda item: int(item.split('/')[-1].split('_')[0])):
            # print(f)
            data_ = pd.read_csv(f, header=None)
            data_ = data_.iloc[:month_size]
            data_.columns = ['x', 'y', 'bed_level', 'depth', 'x_v', 'y_v']
            data_bed.append(data_)
        # for ele in data_bed:
        #     print(ele.iloc[0:5])
        water_df = pd.read_csv(self.DIR_ROOT + "water_level_all.csv")
        water_df['date'] = water_df['date'].map(lambda x: float(x[5:7]))
        water_level = water_df.values[:, :-1]
        print("water_level.shape", water_level.shape, water_level)

        # In[97]:
        data_all = []
        train_end = months - predict_len
        for i in range(0, months):
            print(i)
            data_ = data_bed[i].values
            data_all.append(data_)
            print(data_.shape)
        data_all = np.array(data_all)
        c_stack = water_level[:months, np.newaxis, :]
        c_stack = np.tile(c_stack, (month_size, 1))
        self.data_all = np.concatenate((data_all, c_stack), axis=2)
        data_arr = np.array(data_all[:train_end])
        data_arr = data_arr.reshape(-1, attribute_size)
        print("data_arr.shape", data_arr.shape)
        self.data_arr_mean = data_arr.mean(axis=0)
        self.data_arr_std = data_arr.std(axis=0)
        import gc
        del data_bed
        gc.collect()
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.start = start
        self.end = end
        self.len = (end - start) * month_size
        print("init dataset success")
        self.train_data = []
        for i in range(start, end):
            self.train_data.append(self.get_data(i))


    def __getitem__(self, index):
        i = index // month_size
        j = index % month_size
        train_x, train_y = self.train_data[i]
        step_x = train_x[j]
        step_y = train_y[j]
        step_x = torch.from_numpy(step_x)
        step_y = torch.from_numpy(step_y)
        return step_x, step_y

    def __len__(self):
        return self.len

    def get_data(self, i):
        np_path = self.DIR_ROOT + "attr" + str(attribute_size) + "_step" + str(step_len) + "_predict" + str(
            predict_len) + "_batch" + str(batch_size) + "_numpy" + str(i) + ".npy"
        data_step = []
        # print("i", i)
        if os.path.exists(np_path):
            data_step = np.load(np_path)
            # print("load data_array ", i, " local")
        else:
            for index, row in enumerate(self.data_all[i]):
                data_ = []
                for j in range(0, step_len + predict_len):
                    data_.append(self.data_all[i + j][index])
                data_step.append(data_)
            data_step = np.array(data_step)
            # print("compute data_array ", i)
            np.save(np_path, data_step)
        # print("data_step.shape", data_step.shape)
        x = data_step[:, :-predict_len, :attribute_size]
        y = data_step[:, -predict_len:, :attribute_size]
        train_x = np.nan_to_num(x)
        train_x = np.divide(train_x - self.data_arr_mean, self.data_arr_std)
        train_y = y
        # print("train: ", train_x, train_y)
        # print("shape: ", train_x.shape, train_y.shape, data_arr_mean.shape)
        # print("train after: ", train_x, train_y)
        # print("train_xy.shape: ", train_x.shape, train_y.shape)
        return train_x, train_y



# In[150]:
num_layer = 3


class LSTM(nn.Module):
    def __init__(self, input_size=3, seq_len=2, hidden_layer_size=160, batch_size=360, output_size=1, num_layer=3,
                 kernel_size=63):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.sandconv1d = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size * input_size)
        self.sandrconv1d = nn.ConvTranspose1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)

    def forward(self, input_seq, hidden_cell):
        conv_output = self.conv_high(input_seq)
        # print("input_seq.shape : " + str(input_seq.shape) + ", conv_output shape : " + str(conv_output.shape))
        hidden_cell = [x.permute(1, 0, 2).contiguous() for x in hidden_cell]
        lstm_out, hidden_cell = self.lstm(conv_output, hidden_cell)
        # print("lstm_out shape", lstm_out.shape)
        linear_input = lstm_out[:, -1].squeeze()
        # print("linear_input shape", linear_input.shape)
        linear_output = self.linear(linear_input)
        # print("linear_output_shape", linear_output.shape)
        predict_high = linear_output.reshape(-1, self.output_size, self.input_size)
        # print("predict_high_shape", predict_high.shape)
        dconv_input = predict_high.permute(1, 2, 0)
        # print("dconv_input_shape", dconv_input.shape)
        predictions = self.sandrconv1d(dconv_input).permute(2, 0, 1).squeeze()
        # print("dconv_input predictions shape", dconv_input.shape, predictions.shape)
        return predict_high, predictions

    def conv_high(self, input_seq):
        conv_input = input_seq.permute(1,2,0)
        conv_output = self.sandconv1d(conv_input)
        conv_output = conv_output.permute(2,0,1)
        return conv_output

    def init_hidden(self, batch_size, device):
        return (torch.zeros(size=(batch_size - kernel_size + 1, num_layer, hidden_layer_size),
                            dtype=torch.float64).to(device),
                torch.zeros(size=(batch_size - kernel_size + 1, num_layer, hidden_layer_size),
                            dtype=torch.float64).to(device))


def train(model, device, optimizer, epoch, epoches):
    loss_function = nn.L1Loss(reduction="mean")
    acc_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train()
    batch_count = 0
    for batch_idx, (seq, labels) in loop:
        seq, labels = seq.to(device), labels.to(device)
        batch_count += 1
        optimizer.zero_grad()
        hiddens = model.module.init_hidden(len(seq), DEVICE)
        y_pred_high, y_pred = model(seq, hiddens)
        loss = loss_function(y_pred[:, predict_len - 1, 2], labels[:, predict_len - 1, 2])
        loss.backward()
        acc_loss += loss.item()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch}/{epoches}]')
        loop.set_postfix(loss= acc_loss / (batch_idx + 1))
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()), "batch_count", batch_count, "acc_loss", acc_loss / batch_count)

def val(model, device, test_loader, epoch, DIR_ROOT):
    loss_function = nn.L1Loss(reduction="mean")
    model.eval()
    test_loss = 0
    seq_loss = 0
    result = []
    with torch.no_grad():
        batch_count = 0
        for batch_idx,(seq,labels) in enumerate(test_loader):
            seq,labels = seq.to(device), labels.to(device)
            batch_count += 1
            hiddens = model.module.init_hidden(len(seq), device)
            y_pred_high,y_pred = model(seq, hiddens)
            result.append(torch.cat((labels[:,0,[0,1]],torch.cat((y_pred[:,:,2],labels[:,:,2]), 1)),1).tolist())
            test_loss += loss_function(y_pred[:,predict_len - 1,2], labels[:,predict_len - 1,2])
            seq_loss += loss_function(y_pred[:,:,2], labels[:,:,2])
    #print("test_loader", batch_count, len(test_loader.dataset), test_loss)
    test_loss /= batch_count
    seq_loss /= batch_count
    global min_epoch, min_loss, seq_min_loss
    if test_loss < min_loss:
        min_loss = test_loss
        seq_min_loss = seq_loss
        min_epoch = epoch
    result_array = np.array(result)
    result_array = result_array.reshape(month_size, -1)
    result_df = pd.DataFrame(result_array)
    result_part = result_df.iloc[:,[0,1,predict_len + 1,predict_len * 2 + 1]]
    print(result_part[2470:2480])
    if test_loss < min_loss:
        result_df.to_csv(DIR_ROOT + str(epoch) + "_step" + str(step_len) + "_predict" + str(predict_len) + "_k" + str(kernel_size)  + "_XinjiuWaterTorchSortConv73_all.csv")
        result_part.to_csv(DIR_ROOT + str(epoch) + "_step" + str(step_len) + "_predict" + str(predict_len) + "_k" + str(kernel_size)  +  "_XinjiuWaterTorchSortConv73_part.csv")
        torch.save(model, DIR_ROOT + str(epoch) + "_step" + str(step_len) + "_predict" + str(predict_len) + "_k" + str(kernel_size)  + "_XinjiuWaterTorchSortConv73.pkl")
    print('\nTest loss: Average loss: {:.4f}'.format(test_loss),"seq_loss", seq_loss)
    print("current epoch", epoch, "min_epoch", min_epoch, "min_loss", min_loss, "seq_min_loss", seq_min_loss)
    return result

if __name__ == '__main__':

    DIR_ROOT = "/home/dlf/xinjiu_all/"
    print("args.local_rank", args.local_rank)
    trainDataSet = SandWaterDataSet(0, months - step_len - predict_len - ignore_len)
    trainSampler = DistributedSampler(trainDataSet)
    train_loader = DataLoader(dataset=trainDataSet, batch_size=batch_size, sampler=trainSampler)
    testDataSet = SandWaterDataSet(months - predict_len - step_len, months - predict_len - step_len + 1)
    testLoader = DataLoader(dataset=testDataSet, batch_size=batch_size)
    model = LSTM(attribute_size, step_len, hidden_layer_size, batch_size, predict_len, num_layer, kernel_size).to(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print(model)
    min_epoch = 0
    min_loss = 1000
    seq_min_loss = 1000

    # In[156]:
    time_begin = time.time()
    for epoch in range(1, epoch_size+1):
        train(model, DEVICE, optimizer, epoch, epoch_size)
        if epoch % epoch_step == 0:
            result = val(model, DEVICE, testLoader, epoch, DIR_ROOT)
    time_end = time.time()
    print("total fit cost time: %.2f s" % (time_end - time_begin))