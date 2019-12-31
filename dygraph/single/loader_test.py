import paddle
import paddle.fluid as fluid
import unittest
import numpy as np

batch_size = 32
epoch_num = 2
sample_num = 128

def reader_creator_random_image(height, width):
    def reader():
        for i in range(sample_num):
            data = np.random.uniform(low=0, high=255, size=[height, width])
            yield data
    return reader

with fluid.dygraph.guard():
    batch_py_reader = fluid.io.DataLoader.from_generator(capacity=2)
    user_defined_reader = paddle.batch(reader_creator_random_image(784, 784), batch_size=batch_size)
    batch_py_reader.set_sample_list_generator(
        user_defined_reader,
        places=fluid.core.CPUPlace())

    for epoch in range(epoch_num):
        for i, data in enumerate(batch_py_reader()):
            if i > 1:
                break
            print("pass a batch")
            # empty network
            pass

  
