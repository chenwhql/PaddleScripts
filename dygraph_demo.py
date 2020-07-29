import numpy
import paddle.fluid as fluid

place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    x = numpy.random.random(size=(10, 1)).astype('float32')
    data = fluid.dygraph.to_variable(x)

    linear = fluid.dygraph.Linear(1, 10)
    res = linear(data)
    print(res)
    print(linear.state_dict())