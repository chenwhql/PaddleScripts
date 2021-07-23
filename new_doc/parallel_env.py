import paddle
import paddle.distributed as dist

def train():
    # 1. initialize parallel environment
    dist.init_parallel_env()

    # 2. get current ParallelEnv
    parallel_env = dist.ParallelEnv()
    print("rank: ", parallel_env.rank)
    print("world_size: ", parallel_env.world_size)

    # print result in process 1:
    # rank: 1
    # world_size: 2
    # print result in process 2:
    # rank: 2
    # world_size: 2

if __name__ == '__main__':
    # 1. start by ``paddle.distributed.spawn`` (default)
    dist.spawn(train, nprocs=2)
    # 2. start by ``paddle.distributed.launch``
    # train()