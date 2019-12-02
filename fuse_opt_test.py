#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy
import os

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

x = fluid.layers.data(name='X', shape=[13], dtype='float32')
y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
y_ = fluid.layers.fc(input=x, size=1, act=None)

loss = fluid.layers.square_error_cost(input=y_, label=y)
avg_loss = fluid.layers.mean(loss)
fluid.optimizer.Momentum(
    learning_rate=0.01, 
    momentum=0.9).minimize(avg_loss)

exe.run(fluid.default_startup_program())

build_strategy = fluid.BuildStrategy()
build_strategy.fuse_all_optimizer_ops = True
# fuse_all_reduce_ops_ default true, coalesce_grad_tensor_pass is on
compiled_prog = fluid.compiler.CompiledProgram(
    fluid.default_main_program()).with_data_parallel(
        loss_name=avg_loss.name,
        build_strategy=build_strategy)

# for i in range(10):
x = numpy.random.random(size=(10, 13)).astype('float32')
y = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_prog, 
                    feed={'X': x, 'Y': y}, 
                    fetch_list=[avg_loss.name])
print("avg loss: %.3f" % loss_data[0])
