import paddle
import paddle.distributed as dist
import numpy as np

def do_train():
    dist.init_parallel_env()
    net = paddle.nn.Linear(2, 2)
    net = paddle.DataParallel(net)
    x = paddle.to_tensor(np.random.random(size=(2, 2)).astype('float32'))
    j = []

    y = net(x)

    dist.all_gather(j, y)
    print(j)

def train(world_size=2):
    if world_size > 1:
        dist.spawn(do_train, nprocs=world_size, args=())
    else:
        do_train()


if __name__ == "__main__":
    train(2)
    # do_train()