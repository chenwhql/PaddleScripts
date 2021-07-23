from paddle.fluid.dygraph.base import to_variable
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
import numpy as np

data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    conv2d = Conv2D(3, 2, 3, act = 'hard_sigmoid')
    # conv2d = Conv2D(3, 2, 3)
    data = to_variable(data)
    conv = conv2d(data)
