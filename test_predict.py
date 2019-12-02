#!/usr/bin/env python
# coding=utf-8
import paddle
import paddle.fluid as fluid
import numpy as np

place = fluid.CPUPlace()
exe = fluid.Executor(place)
main_program = fluid.default_main_program()
    
x = fluid.layers.data(name='x', shape=[256], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

batchnum = 5
batch_data = [[np.random.rand(256)] for a in range(batchnum)]
    
feeder = fluid.DataFeeder(place=place, feed_list=[x])
#exe.run(fluid.default_startup_program())
output = exe.run(main_program,
                 feed=feeder.feed(batch_data),
                 fetch_list=[y_predict])
print(output)
