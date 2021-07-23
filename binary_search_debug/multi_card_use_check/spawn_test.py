import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist
from paddle.static.input import InputSpec

import os


def multi_gpus_used():
    res = os.popen("nvidia-smi --query-gpu=memory.used --format=csv").read()
    lines = res.splitlines()
    memory_use = []
    for line in lines[2:4]:
        units = line.split(' ')
        memory_use.append(int(units[0]))
    for mem in memory_use:
        if mem <= 10:
            return False
    return True


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)
        
    def forward(self, x):
        return self._linear2(self._linear1(x))

def train():
    # initialize parallel environmen
    dist.init_parallel_env()

    print("chenweihang: init parallel env")

    # create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters())

    # 4. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("chenweihang: forward")
    
    loss.backward()

    print("chenweihang: backward")

    error = multi_gpus_used()

    adam.step()
    adam.clear_grad()

    # print whether use multi gpu
    if dist.get_rank() == 0:
        print(error)


if __name__ == '__main__':
    dist.spawn(train, nprocs=2)