#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy
import os
import paddle.fluid.transpiler.details.program_utils as pu

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

x = fluid.layers.data(name='X', shape=[13], dtype='float32')
y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
y_ = fluid.layers.fc(input=x, size=1, act=None)

loss = fluid.layers.square_error_cost(input=y_, label=y)
avg_loss = fluid.layers.mean(loss)

fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

exe.run(fluid.default_startup_program())

print("startup program:")
print(fluid.default_startup_program())
# pu.program_to_code(fluid.default_startup_program(), skip_op_callstack=True)
print("main program:")
print(fluid.default_main_program())
# pu.program_to_code(fluid.default_main_program(), skip_op_callstack=True)

x = numpy.random.random(size=(10, 13)).astype('float32')
y = numpy.random.random(size=(10, 1)).astype('float32')

compiled_prog = fluid.compiler.CompiledProgram(
        fluid.default_main_program())

for step in range(0, 1):
    loss_data, = exe.run(
        compiled_prog, 
        feed={'X': x, 'Y': y}, 
        fetch_list=[avg_loss.name])
    print("loss:")
    print(loss_data[0])
