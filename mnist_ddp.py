import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import os
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from resnet import BasicBlock, Bottleneck, ResNet, resnet152

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--local_rank", type=int)  # 增加local_rank
parser.add_argument('--epoch_size', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
epochs = args.epoch_size
batch_size = args.batch_size
dist.init_process_group("nccl", init_method='env://')
device = args.local_rank
DEVICE = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
val_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


def Train(cnn, loaders):
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])
    train_loss = .0
    for i, (images, labels) in enumerate(loaders['train']):
        b_x = Variable(images).cuda()  # batch x
        b_y = Variable(labels).cuda()  # batch y
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
    train_loss /= total_step
    print('Loss: {}'.format(train_loss.item()))


def Test():
    cnn.eval()
    with torch.no_grad():
        accuracy = 0
        total_step = len(loaders['test'])
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images.cuda())
            pred_y = torch.max(test_output.cpu(), 1)[1].data.squeeze()
            accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
        accuracy /= total_step
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


if __name__ == '__main__':
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
    train_sampler = DistributedSampler(train_data)
    test_sampler = DistributedSampler(test_data)
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             sampler=train_sampler,
                                             num_workers=8),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            sampler=test_sampler,
                                            num_workers=8),
    }
    cnn = CNN().cuda()
    cnn = nn.parallel.DistributedDataParallel(cnn, device_ids=[args.local_rank],
                                              output_device=args.local_rank)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    for epoch in range(epochs):
        Train(cnn, loaders)
        Test()
