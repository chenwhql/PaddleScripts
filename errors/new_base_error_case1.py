#!/usr/bin/env python
# coding=utf-8
import numpy
import paddle
import paddle.static as static
import paddle.nn.functional as F

# 开启静态图模式
paddle.enable_static()
paddle.set_device('cpu')

# 网络结构定义
x = static.data(name='X', shape=[None, 13], dtype='float32')
y = static.data(name='Y', shape=[None, 1], dtype='float32')
predict = static.nn.fc(x=x, size=1)
loss = F.square_error_cost(input=predict, label=y)
avg_loss = paddle.mean(loss)

# 执行环境准备
exe = static.Executor(paddle.CPUPlace())
exe.run(static.default_startup_program())

# 执行网络
x = numpy.random.random(size=(7, 13)).astype('float32')
y = numpy.random.random(size=(8, 1)).astype('float32')
loss_data, = exe.run(static.default_main_program(), feed={'X': x, 'Y': y}, fetch_list=[avg_loss.name])
