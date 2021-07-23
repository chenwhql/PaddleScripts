from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

def train(print_result=False):
    # 1. enable dynamic mode
    paddle.disable_static()
    
    # 2. initialize parallel environment
    dist.init_parallel_env()

    # 3. create data parallel layer & optimizer
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
    
    if print_result is True:
        print("loss:", loss.numpy())
    
    loss.backward()

    adam.step()
    adam.clear_grad()

# Usage 1: only pass function. 
# If your training method no need any argument, and 
# use all visible devices for parallel training. 
if __name__ == '__main__':
    dist.spawn(train)

# Usage 2: pass function and arguments.
# If your training method need some arguments, and 
# use all visible devices for parallel training.
if __name__ == '__main__':
    dist.spawn(train, args=(True,))

# Usage 3: pass function, arguments and nprocs.
# If your training method need some arguments, and 
# only use part of visible devices for parallel training.
# If your machine hold 8 cards {0,1,2,3,4,5,6,7},
# this case will use cards {0,1}; If you set 
# CUDA_VISIBLE_DEVICES=4,5,6,7, this case will use
# cards {4,5}
if __name__ == '__main__':
    dist.spawn(train, args=(True,), nprocs=2)

# Usage 4: pass function, arguments, nprocs and selected_gpus.
# If your training method need some arguments, and 
# only use part of visible devices for parallel training,
# but you can't set your machine's environment varibale 
# CUDA_VISIBLE_DEVICES, such as it is None or all cards
# {0,1,2,3,4,5,6,7}, you can pass `selelcted_gpus` to 
# select the GPU cards you want to use. For example,
# this case will use cards {4,5} if your machine hold 8 cards.
if __name__ == '__main__':
    dist.spawn(train, args=(True,), nprocs=2, selelcted_gpus='4,5')