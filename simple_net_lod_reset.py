#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy
import os

place = fluid.CPUPlace()
exe = fluid.Executor(place)

x = fluid.data(name='X', shape = [None, 13], dtype = 'float32')
print(x)
x = fluid.layers.lod_reset(x=x, target_lod=[0, 3])
print(x)

y = fluid.data(name='Y', shape=[None, 1], dtype='float32')
y_ = fluid.layers.fc(input=x, size=1, act=None)

loss = fluid.layers.square_error_cost(input=y_, label=y)
avg_loss = fluid.layers.mean(loss)

fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

exe.run(fluid.default_startup_program())

x = numpy.random.random(size=(10, 13)).astype('float32')
y = numpy.random.random(size=(10, 1)).astype('float32')

compiled_prog = fluid.compiler.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(
            loss_name=avg_loss.name, places=fluid.CPUPlace())

for step in range(0, 1):
    loss_data, = exe.run(
        compiled_prog, 
        feed={'X': x, 'Y': y}, 
        fetch_list=[avg_loss])
    print("loss:", loss_data)
