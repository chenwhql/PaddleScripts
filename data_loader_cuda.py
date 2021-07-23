import paddle
import multiprocessing

mp = multiprocessing.get_context('spawn')

paddle.set_device('gpu')
t = paddle.randn(shape=[2, 3], dtype="float32")

data_queue = mp.Queue()
data_queue.put(t)

def get_cuda_tensor(data_queue):
    t = data_queue.get()
    print(t)

worker = mp.Process(
    target=get_cuda_tensor,
    args=(data_queue))
worker.start()
