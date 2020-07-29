import os
import paddle
import paddle.fluid as fluid
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# def reader(exe, program, feed, fetch):
def reader():
    def __impl__():
      place = fluid.CUDAPlace(0)
      # place = fluid.CPUPlace()

      img = fluid.data(name='img', shape=[None, 784], dtype='float32')
      label = fluid.data(name='label', shape=[None, 1], dtype='int64')
      pred = fluid.layers.fc(input=img, size=10, act='softmax')

      exe = fluid.Executor(place)
      exe.run(fluid.default_startup_program())
      tensor_img = np.array(np.random.random((1, 784)), dtype=np.float32)
      results = exe.run(program,
                        feed={feed[0]: tensor_img},
                        fetch_list=fetch)
      yield results[0]

    return __impl__
if __name__=="__main__":
    # place = fluid.CUDAPlace(0)
    # # place = fluid.CPUPlace()

    # img = fluid.data(name='img', shape=[None, 784], dtype='float32')
    # label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    # pred = fluid.layers.fc(input=img, size=10, act='softmax')

    # exe = fluid.Executor(place)
    # exe.run(fluid.default_startup_program())

    # execute_func = reader(exe, fluid.default_main_program(), ['img'], [pred])
    execute_func = reader()
    # train_reader = execute_func
    train_reader = paddle.reader.multiprocess_reader([execute_func], use_pipe=False)
    for data in train_reader():
        print(data)