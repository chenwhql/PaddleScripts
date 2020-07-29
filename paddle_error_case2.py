import numpy
import paddle.fluid as fluid

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    x = numpy.random.random(size=(10, 1)).astype('float32')
    linear = fluid.dygraph.Linear(1, 10)
    data = fluid.dygraph.to_variable(x)
    res = linear(data)