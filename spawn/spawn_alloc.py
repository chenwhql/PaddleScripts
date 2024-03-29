import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist
from paddle.static.input import InputSpec

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

def train():
    # 1. enable dynamic mode
    paddle.disable_static()

    # 2. initialize parallel environmen
    dist.init_parallel_env()

    # 3. create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters())

    # 4. run layer
    for i in range(100000):
        inputs = paddle.randn([10, 10], 'float32')
        outputs = dp_layer(inputs)
        labels = paddle.randn([10, 1], 'float32')
        loss = loss_fn(outputs, labels)

        loss.backward()

        adam.step()
        adam.clear_grad()

        if dist.get_rank() == 0:
            print("loss:", loss.numpy())

if __name__ == '__main__':
    dist.spawn(train, nprocs=4)