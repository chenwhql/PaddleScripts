#!/usr/bin/env python
# coding=utf-8
import paddle.fluid as fluid

x=fluid.layers.data(name='x', append_batch_size = False, shape=[2, 5, 4, 3], dtype='float64')
y=fluid.layers.data(name='y', append_batch_size = False, shape=[2, 4, 4, 3], dtype='float64')
mul=fluid.layers.mul(x=x, y=y, x_num_col_dims=2, y_num_col_dims=2)
