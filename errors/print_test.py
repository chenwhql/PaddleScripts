import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize

EPOCH_NUM = 1
BATCH_SIZE = 64

class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x



paddle.enable_static()
net = LeNet()
x = paddle.static.data("x", [None, 1, 28, 28], "float32")
out = net(x)

# print(paddle.static.default_main_program())
paddle.save(paddle.static.default_main_program(), "./lenet.pdmodel")

# net = LeNet()
# print(net)