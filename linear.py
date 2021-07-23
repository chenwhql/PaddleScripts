from paddle.fluid.dygraph.base import to_variable
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np

data = np.random.uniform( -1, 1, [30, 10, 32] ).astype('float32')
with fluid.dygraph.guard():
    linear = Linear(32, 64)
    data = to_variable(data)
    res = linear(data)  # [30, 10, 64]