import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
from paddle.vision.transforms import Compose, Normalize
from paddle.utils.cpp_extension import load

EPOCH_NUM = 4
BATCH_SIZE = 64

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])


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
        x = custom_ops.custom_relu(x)
        x = self.max_pool1(x)
        x = custom_ops.custom_relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = custom_ops.custom_relu(x)
        x = self.linear2(x)
        x = custom_ops.custom_relu(x)
        x = self.linear3(x)
        return x


# set device
paddle.enable_static()
paddle.set_device("gpu")

# model
image  = static.data(shape=[None, 1, 28, 28], name='image', dtype='float32')
label = static.data(shape=[None, 1], name='label', dtype='int64')

net = LeNet()
out = net(image)
loss = nn.functional.cross_entropy(out, label)

opt = paddle.optimizer.Adam(learning_rate=0.001)
opt.minimize(loss)

# data loader
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset,
    feed_list=[image, label],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# prepare
exe = static.Executor()
exe.run(static.default_startup_program())

places = paddle.static.cuda_places()
compiled_program = static.CompiledProgram(
    static.default_main_program()).with_data_parallel(
        loss_name=loss.name, places=places)

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image_data, label_data) in enumerate(train_loader()):
        loss_data = exe.run(compiled_program,
            feed={'image': image_data,
                  'label': label_data},
            fetch_list=[loss])
        if batch_id % 300 == 0:
            print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss_data)))
    
# save inference model
path = "custom_relu_test_static/net"
static.save_inference_model(path, [image], [out], exe)