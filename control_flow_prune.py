#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy
import os

paddle.enable_static()

place = fluid.CPUPlace()
exe = fluid.Executor(place)

x = fluid.data(name='X', shape=[1], dtype = 'float32')
y = fluid.data(name='Y', shape=[1], dtype='float32')
z = fluid.data(name="Z", shape=[1], dtype='bool')

out = fluid.layers.cond(z, lambda: x, lambda: y)

exe.run(fluid.default_startup_program())

print(fluid.default_main_program())

x = numpy.random.random(size=(1)).astype('float32')
y = numpy.random.random(size=(1)).astype('float32')
z = True

# out = exe.run(
#       fluid.default_startup_program(), 
#       feed={'X': x, 'Y': y, 'Z': z}, 
#       fetch_list=[out])

exe.run(
      fluid.default_startup_program(), 
      feed={'X': x, 'Y': y, 'Z': z})

fluid.io.save_inference_model(
    dirname="./control_flow_prune",
    feeded_var_names=['X', 'Z'],
    target_vars=[out],
    executor=exe)

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
    dirname="./control_flow_prune",
    executor=exe)

print(inference_program)
