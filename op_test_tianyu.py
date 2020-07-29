#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file optest
  * @author zhengtianyu
  * @date 2020/6/5 4:36 下午
  * @brief 
  *
  **************************************************************************/
"""
from paddle.fluid.dygraph.base import to_variable
import paddle.fluid as fluid
import numpy as np

class Model(fluid.dygraph.Layer):
    """
    test model
    """
    def __init__(self, func):
        super(Model, self).__init__()
        self.layer = func
        self.linear = fluid.dygraph.Linear(6, 3, act="softmax")

    def forward(self, input):
        x = self.layer(input)
        print(x.numpy())
        res = self.linear(x)
        return res


def dygraph():
    """
    test dygraph
    :return:
    """
    with fluid.dygraph.guard():
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        m = Model(fluid.dygraph.Linear(6, 6, act="relu"))
        opt = fluid.optimizer.SGDOptimizer(0.001, parameter_list=m.parameters())
        for i in range(2):
            # basic
            data = np.random.random(size=(3, 6)).astype(np.float32)
            label = np.random.randint(low=0, high=2, size=(3, 1))
            data = to_variable(data)
            label = to_variable(label)
            res = m(data)
            loss = fluid.layers.cross_entropy(res, label)
            avg = fluid.layers.mean(loss)
            avg.backward()
            print(avg.numpy())
            opt.minimize(loss)



def dygraph1():
    """
    test dygraph1
    :return:
    """
    with fluid.dygraph.guard():
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        for i in range(2):
            # basic
            data = np.random.random(size=(3, 6)).astype(np.float32)
            label = np.random.randint(low=0, high=2, size=(3, 1))
            data = to_variable(data)
            label = to_variable(label)
            l1 = fluid.dygraph.Linear(6, 6, act="relu")
            l2 = fluid.dygraph.Linear(6, 3, act="softmax")
            a = l1.parameters() + l2.parameters()
            opt = fluid.optimizer.SGDOptimizer(0.001, parameter_list=a)
            d1 = l1(data)
            print(d1.numpy())
            res = l2(d1)
            loss = fluid.layers.cross_entropy(res, label)
            avg = fluid.layers.mean(loss)
            avg.backward()
            print(avg.numpy())
            opt.minimize(loss)
            l1.clear_gradients()
            l2.clear_gradients()


dygraph()

print("===========")
dygraph1()