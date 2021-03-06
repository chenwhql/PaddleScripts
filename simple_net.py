#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy
import os

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

x = fluid.data(name='X', shape = [None, 13], dtype = 'float32')
y = fluid.data(name='Y', shape=[None, 1], dtype='float32')
y_ = fluid.layers.fc(input=x, size=1, act=None)

loss = fluid.layers.square_error_cost(input=y_, label=y)
avg_loss = fluid.layers.mean(loss)

fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

exe.run(fluid.default_startup_program())

print("startup program:")
print(fluid.default_startup_program())
print("main program:")
print(fluid.default_main_program())

print(fluid.default_main_program().clone(for_test=True))

x = numpy.random.random(size=(10, 13)).astype('float32')
y = numpy.random.random(size=(10, 1)).astype('float32')

fetch_list = ['square_error_cost_0.tmp_1', 'mean_0.tmp_0', 'mean_0.tmp_0@GRAD', 'square_error_cost_0.tmp_1@GRAD']

compiled_prog = fluid.compiler.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(
            loss_name=avg_loss.name)

for step in range(0, 1):
    loss_data, avg_loss_data, avg_loss_grad, loss_grad = exe.run(
        compiled_prog, 
        feed={'X': x, 'Y': y}, 
        fetch_list=fetch_list)
    print("loss:")
    print(loss_data)
    print("avg_loss: %.3f, avg_loss_grad: %.3f" 
        % (avg_loss_data[0], avg_loss_grad[0]))
    print("loss_grad")
    print(loss_grad)
