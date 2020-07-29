import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import unittest

from paddle.fluid.dygraph.jit import declarative


def static_func(x):
    x = fluid.layers.assign(x)
    iter_num = fluid.layers.fill_constant(shape=[1], value=3, dtype='int32')
    a = fluid.layers.create_array(dtype='float32')
    i = 0
    a = fluid.dygraph.dygraph_to_static.variable_trans_func.to_static_variable(
        a)
    i = fluid.dygraph.dygraph_to_static.variable_trans_func.to_static_variable(
        i)
    iter_num = (fluid.dygraph.dygraph_to_static.variable_trans_func.
                to_static_variable(iter_num))
    x = fluid.dygraph.dygraph_to_static.variable_trans_func.to_static_variable(
        x)

    def while_condition_0(a, i, iter_num, x):
        return i < iter_num

    def while_body_0(a, i, iter_num, x):
        fluid.layers.array_write(x=x, i=fluid.layers.array_length(a), array=a)
        i += 1
        return a, i, iter_num, x
    a, i, iter_num, x = fluid.layers.while_loop(while_condition_0,
                                                while_body_0, [a, i, iter_num, x])
    length = layers.array_length(a)
    layers.Print(length)
    return a[0]


@declarative
def dygraph_func(x):
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=3, dtype="int32")
    a = []
    i = 0
    while i < iter_num:
        a.append(x)
        i += 1
    length = layers.array_length(a)
    layers.Print(length)
    return a[0]

g_scope = core.Scope()
place = fluid.CPUPlace()
exe = fluid.Executor(place)
main_program = fluid.Program()

# with fluid.program_guard(main_program):
#     input = np.random.random((3)).astype('int32')
#     static_func(input)
#     for i in range(3):
#         exe.run(main_program, scope=g_scope)


with fluid.dygraph.guard():
    input = np.random.random((3)).astype('int32')
    for i in range(2):
        res = dygraph_func(input)
