from __future__ import print_function

import unittest
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward
import numpy

import paddle.fluid.transpiler.details.program_utils as pu

def simple_net():
    d0 = layers.data(
        "d0", shape=[10], append_batch_size=False, dtype='float32')
    d1 = layers.data(
        "d1", shape=[10], append_batch_size=False, dtype='float32')
    d2 = layers.data(
        "d2", shape=[10], append_batch_size=False, dtype='float32')
    i = layers.zeros(shape=[1], dtype='int64')
    i.stop_gradient = True
    init = layers.zeros(shape=[10], dtype='float32')
    mem_array = layers.array_write(x=init, i=i)
    data_array = layers.array_write(x=d0, i=i)
    i = layers.increment(i)
    layers.array_write(d1, i, array=data_array)
    i = layers.increment(i)
    layers.array_write(d2, i, array=data_array)
    i = layers.zeros(shape=[1], dtype='int64')
    i.stop_gradient = True
    array_len = layers.fill_constant(shape=[1], dtype='int64', value=1)
    array_len.stop_gradient = True
    cond = layers.less_than(x=i, y=array_len)
    j = layers.fill_constant(shape=[1], dtype='int64', value=1)
    j.stop_gradient = True
    array_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
    array_len2.stop_gradient = True
    cond2 = layers.less_than(x=j, y=array_len2)
    while_op = layers.While(cond=cond)
    while_op2 = layers.While(cond=cond2)
    with while_op.block():
        d = layers.array_read(array=data_array, i=i)
        prev = layers.array_read(array=mem_array, i=i)
        result = layers.sums(input=[d, prev])

        i = layers.increment(x=i, in_place=True)
        layers.array_write(result, i=i, array=mem_array)
        layers.less_than(x=i, y=array_len, cond=cond)

        with while_op2.block():
            d2 = layers.array_read(array=data_array, i=j)
            prev2 = layers.array_read(array=mem_array, i=j)
            result2 = layers.sums(input=[d2, prev2])

            j = layers.increment(x=j, in_place=True)
            layers.array_write(result2, i=j, array=mem_array)
            layers.less_than(x=j, y=array_len2, cond=cond2)
    sum_result = layers.array_read(array=mem_array, i=j)
    loss = layers.mean(sum_result)
    return loss, sum_result

main_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(main_program, startup_program):
    loss, sum_result = simple_net()

    append_backward(loss)

    cpu = core.CPUPlace()
    exe = Executor(cpu)
    d = []

    # pu.program_to_code(fluid.default_main_program(), skip_op_callstack=True)

    for i in range(3):
        d.append(numpy.random.random(size=[10]).astype('float32'))

    outs = exe.run(feed={'d0': d[0],
                         'd1': d[1],
                         'd2': d[2]},
                   fetch_list=[sum_result])

    fluid.io.save_inference_model(
                "control_flow.while", ['d0', 'd1', 'd2'], [sum_result],
                exe,
                main_program=fluid.default_main_program(),
                model_filename=None,
                params_filename=None)

    #####

    [program, feed_target_names, fetch_targets] = \
        fluid.io.load_inference_model(
             "control_flow.while", exe, model_filename=None, params_filename=None)

    pu.program_to_code(program, skip_op_callstack=True)
    
    print("feed_target_names:")
    print(feed_target_names)
    print("fetch_targets:")
    print(fetch_targets)
