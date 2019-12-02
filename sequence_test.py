#!/usr/bin/env python
# coding=utf-8
import numpy
import paddle.fluid as fluid

x = fluid.layers.data(name='x', shape=[10,10], append_batch_size=False, dtype='float32')
x_conved = fluid.layers.sequence_last_step(input=x)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

input = numpy.random.random([10,10]).astype('float32')
exe.run(fluid.default_main_program(), feed={'x': input})
