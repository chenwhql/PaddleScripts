#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy
import os

before_optimize = True

place = fluid.CPUPlace()
exe = fluid.Executor(place)

x = fluid.layers.data(name='X', shape=[13], dtype='float32')
y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
y_ = fluid.layers.fc(input=x, size=1, act=None)

loss = fluid.layers.square_error_cost(input=y_, label=y)
avg_loss = fluid.layers.mean(loss)

if before_optimize:
    exe.run(fluid.default_startup_program())

fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

if not before_optimize:
    exe.run(fluid.default_startup_program())

x = numpy.random.random(size=(10, 13)).astype('float32')
y = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(fluid.default_main_program(), feed={'X': x, 'Y': y}, fetch_list=[avg_loss.name])

print("avg loss: %.3f" % loss_data[0])
