import paddle
import paddle.fluid as fluid
import numpy

fluid.enable_dygraph()

x = numpy.random.random(size=(10, 1)).astype('float32')
data1 = fluid.dygraph.to_variable(x)

x = numpy.random.random(size=(10, 1)).astype('float32')
data2 = fluid.dygraph.to_variable(x)

test_list = []
test_list.append(data1)
test_list.append(None)
test_list.append(None)
test_list.append(data2)

def is_not_none(x):
    return x is not None

input_var = list(filter(is_not_none, test_list))

print(input_var)