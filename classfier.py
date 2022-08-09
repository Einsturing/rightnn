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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
epochs = 1000
batch_size = 32


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
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
        self.out = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


def Train(model, loaders):
    model.train()

    # Train the model
    total_step = len(loaders['train'])
    train_loss = .0
    for i, (images, labels) in enumerate(loaders['train']):
        b_x = Variable(images).cuda()  # batch x
        b_y = Variable(labels).cuda()  # batch y
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
    train_loss /= total_step
    print('Loss: {}'.format(train_loss.item()))


def Test():
    model.eval()
    with torch.no_grad():
        accuracy = 0
        total_step = len(loaders['test'])
        for images, labels in loaders['test']:
            test_output = model(images.cuda())
            pred_y = torch.max(test_output.cpu(), 1)[1].data.squeeze()
            accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
        accuracy /= total_step
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


if __name__ == '__main__':
    # train_data_mnist = datasets.MNIST(
    #     root='data',
    #     train=True,
    #     transform=ToTensor(),
    #     download=True,
    # )
    # test_data_mnist = datasets.MNIST(
    #     root='data',
    #     train=False,
    #     transform=ToTensor()
    # )
    train_data = datasets.CIFAR10(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        transform=ToTensor(),
        download=True,
    )
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             num_workers=8),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            num_workers=8),
    }
    # model = CNN().cuda()
    model = resnet152().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        print(epoch)
        Train(model, loaders)
        Test()
