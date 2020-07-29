import numpy
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative

@declarative
def mul(a, b):
    print('Tracing:\n    {a}\n    {b}\n'.format(a=a, b=b))
    return a*b

with fluid.dygraph.guard():
    a = fluid.layers.fill_constant([1], "int32", 2)
    b = fluid.layers.fill_constant([1], "int32", 3)
    print(mul(a, b).numpy())

    a = fluid.layers.fill_constant([1], "float32", 2.5)
    b = fluid.layers.fill_constant([1], "float32", 3.5)
    print(mul(a, b).numpy())