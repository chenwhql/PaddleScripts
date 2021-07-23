import numpy as np
import paddle
import paddle.nn as nn
from paddle.utils.cpp_extension import load

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class Net(nn.Layer):
    """
    A simple example for Regression Model.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, 100)
        self.fc2 = nn.Linear(100, CLASS_NUM)

    def forward(self, x):
        tmp1 = self.fc1(x)
        # call custom relu op
        tmp_out = custom_ops.custom_relu(tmp1)
        tmp2 = self.fc2(tmp_out)
        # call custom relu op
        out = custom_ops.custom_relu(tmp2)
        return out

paddle.set_device("gpu")

# create network
net = Net()
loss_fn = nn.CrossEntropyLoss()
opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image, label) in enumerate(loader()):
        out = net(image)
        loss = loss_fn(out, label)
        loss.backward()
        
        opt.step()
        opt.clear_grad()
        print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

# save inference model
path = "custom_relu_test_dynamic/net"
paddle.jit.save(net, path,
    input_spec=[paddle.static.InputSpec(shape=[None, 784], dtype='float32')])