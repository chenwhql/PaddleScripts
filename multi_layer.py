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


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)

def train():
    dist.init_parallel_env()
    # 1. initialize parallel environment
    set_seed(2021)
    # 2. create data parallel layer & optimizer
    layer = LinearNet()

    if dist.get_world_size() > 1:
        dp_layer = paddle.DataParallel(layer)
    else:
        dp_layer = layer

    layer2 = LinearNet()

    if dist.get_world_size() > 1:
        dp_layer2 = paddle.DataParallel(layer2)
    else:
        dp_layer2 = layer2

    dp_layer2.set_state_dict(dp_layer.state_dict())

    loss_fn = nn.MSELoss()
    adam = opt.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters())

    adam2 = opt.Adam(
        learning_rate=0.001, parameters=dp_layer2.parameters())
    # 3. run layer

    print("Start")
    for i in range(10):
        batch_size = 10
        shard = int(batch_size / dist.get_world_size())
        start_no = shard * dist.get_rank()
        end_no = start_no + shard
        inputs = paddle.randn([10, 10], 'float32')[start_no:end_no]
        outputs = dp_layer(inputs)
        labels = paddle.randn([10, 1], 'float32')[start_no:end_no]
        loss = loss_fn(outputs, labels)
        if dist.get_rank() == 0:
            print("Loss1", loss.numpy()[0])
            print(dp_layer.parameters())
        loss.backward()
        adam.step()
        adam.clear_grad()

        outputs = dp_layer2(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        if dist.get_rank() == 0:
            print("Loss2", loss.numpy()[0])
            print(dp_layer2.parameters())
        adam2.step()
        adam2.clear_grad()

if __name__ == '__main__':
    train()
