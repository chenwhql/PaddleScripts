import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist
import numpy as np

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

class FakeDataset(paddle.io.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        # return paddle.randn([10, 10], 'float32'), paddle.randn([10, 1], 'float32')
        return np.random.random([10, 10]).astype('float32'), np.random.random([10, 1]).astype('float32')

    def __len__(self):
        return 8

def train(print_result=True):
    # 1. enable dynamic mode
    # device = paddle.set_device('gpu')
    # paddle.disable_static(device)

    # 2. initialize parallel environment
    dist.init_parallel_env()

    # 3. create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters())

    dataset = FakeDataset()
    # loader = paddle.io.DataLoader(dataset, batch_size=2, places=device, num_workers=2)
    loader = paddle.io.DataLoader(dataset, batch_size=2, num_workers=2)
    # 4. run layer
    for inputs, labels in loader:
        # inputs = paddle.randn([10, 10], 'float32')
        outputs = dp_layer(inputs)
        # labels = paddle.randn([10, 1], 'float32')
        loss = loss_fn(outputs, labels)

        if print_result is True:
            print("loss:", loss.numpy())

        # loss = dp_layer.scale_loss(loss)
        loss.backward()
        # dp_layer.apply_collective_grads()

        adam.step()
        adam.clear_grad()

# Usage 1: only pass function.
# If your training method no need any argument, and
# use all visible devices for parallel training.
if __name__ == '__main__':
    # train()
    dist.spawn(train)