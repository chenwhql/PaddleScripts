from __future__ import print_function

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.jit import dygraph_to_static_func

def dyfunc_with_if_else(x_v, label=None):
    if fluid.layers.mean(x_v).numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    # plain if in python
    if label is not None:
        loss = fluid.layers.cross_entropy(x_v, label)
        return loss
    return x_v

main_program = fluid.Program()
with fluid.program_guard(main_program):
    x = np.random.random([5, 6]).astype('float32')
    x_v = fluid.layers.assign(x)
    # Transform into static graph
    out = dygraph_to_static_func(dyfunc_with_if_else)(x_v)
    exe = fluid.Executor(fluid.CPUPlace())
    ret = exe.run(main_program, fetch_list=out)
    print(ret[0])