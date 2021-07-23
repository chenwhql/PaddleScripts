import paddle
import paddle.distributed as dist

def train():
  #print("paddle.distributed.ParallelEnv().dev_id:", paddle.distributed.ParallelEnv().device_id)
  #paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
  #paddle.set_device('gpu:0')
  #with paddle.fluid.dygraph.guard(paddle.fluid.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)):
  #dist.init_parallel_env()
  print("paddle.distributed.ParallelEnv().dev_id:", paddle.distributed.ParallelEnv().device_id)
  paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().device_id)
  #place = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
  #paddle.disable_static(place)
  print("paddle.get_device()", paddle.get_device())
  #print("paddle.distributed.ParallelEnv().dev_id:", paddle.distributed.ParallelEnv().dev_id)
  state = paddle.load("./fc.example.model")
  print(state.keys().__len__())

if __name__ == "__main__":
  #train()
  dist.spawn(train, nprocs=2)