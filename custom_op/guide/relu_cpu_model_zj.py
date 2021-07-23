  
import numpy as np
import paddle
import paddle.nn as nn
from paddle.vision.transforms import ToTensor
from paddle.utils.cpp_extension import load

BATCH_SIZE = 32
EPOCH_NUM = 10

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cpu_float32.cc"])

transform = ToTensor()
cifar10_train = paddle.vision.datasets.Cifar10(mode='train',
                                               transform=transform)


class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(MyNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3))

        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = custom_ops.custom_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = custom_ops.custom_relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = custom_ops.custom_relu(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = custom_ops.custom_relu(x)
        x = self.linear2(x)
        return x

#paddle.set_device("gpu")
paddle.set_device("cpu")

# create network
model = MyNet(num_classes=10)
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())


# create data loader
train_loader = paddle.io.DataLoader(cifar10_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
for epoch in range(EPOCH_NUM):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = paddle.to_tensor(data[1])
        y_data = paddle.unsqueeze(y_data, 1)

        logits = model(x_data)
        loss = paddle.nn.functional.cross_entropy(logits, y_data)

        if batch_id % 1000 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
        loss.backward()
        opt.step()
        opt.clear_grad()