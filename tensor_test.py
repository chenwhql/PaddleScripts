import numpy as np
import paddle.fluid as fluid

place = fluid.CPUPlace()
data = np.random.random([16]).astype('float32')
res = fluid.Tensor()
res.set(data, place)
print(res.shape()[0])