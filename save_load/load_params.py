import paddle
import paddle.distributed as dist

def train():
    # dev_id = dist.ParallelEnv().dev_id
    # print(dev_id)
    # place = paddle.CUDAPlace(dev_id)
    place = paddle.CPUPlace()
    paddle.disable_static(place)
    state, _ = paddle.load("./fc.inference.model")
    print(state.keys())

if __name__ == "__main__":
    train()
    #dist.spawn(train, nprocs=2)