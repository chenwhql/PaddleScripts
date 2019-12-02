#!/usr/bin/env python
# coding=utf-8

import paddle.fluid as fluid

place = fluid.CPUPlace()
exe = fluid.Executor(place)

x = fluid.layers.data(name='X', shape=[13], dtype='float32')
y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
y_ = fluid.layers.fc(input=x, size=1, act=None)
loss = fluid.layers.square_error_cost(input=y_, label=y)
avg_loss = fluid.layers.mean(loss)
avg_loss.persistable = True

fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

exe.run(fluid.default_startup_program())
fluid.io.save_inference_model(dirname="./save_op_test",
                              feeded_var_names=['X', 'Y'],
                              target_vars=[avg_loss],
                              executor=exe)