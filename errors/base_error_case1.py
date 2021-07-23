#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy

paddle.enable_static()

# 1. 网络结构定义
x = fluid.layers.data(name='X', shape=[-1, 13], dtype='float32')
y = fluid.layers.data(name='Y', shape=[-1, 1], dtype='float32')
predict = fluid.layers.fc(input=x, size=1, act=None)
loss = fluid.layers.square_error_cost(input=predict, label=y)
avg_loss = fluid.layers.mean(loss)

# 2. 优化器配置
fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

# 3. 执行环境准备
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 4. 执行网络
x = numpy.random.random(size=(8, 12)).astype('float32')
y = numpy.random.random(size=(8, 1)).astype('float32')
loss_data, = exe.run(fluid.default_main_program(), feed={'X': x, 'Y': y}, fetch_list=[avg_loss.name])
