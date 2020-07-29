#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file test_dataloader
  * @author zhengtianyu
  * @date 2020/1/22 2:48 下午
  * @brief
  *
  **************************************************************************/
"""
import paddle.fluid as fluid
import paddle
import time
import numpy as np

import logging


def reader_decorator(reader):
    """
    reader_decorator
    :param reader:
    :return:
    """
    def __reader__():
        """
        __reader__
        :return:
        """
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(1, 28, 28)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label
    return __reader__


def next_sample(reader):
    try:
        sample = next(reader)
    except StopIteration:
        sample = None
    return sample


def test_multi_process_data_loader():
    """
    test multi process data loader
    :return:
    """
    with fluid.dygraph.guard():
        train_reader = paddle.batch(
                    reader_decorator(
                        paddle.dataset.mnist.train()),
                        batch_size=1,
                        drop_last=True)
        train_loader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
        train_loader.set_sample_list_generator(train_reader, places=fluid.CPUPlace())
        a = list(train_reader())
        b = list(train_loader())
        assert len(a) == len(b)
        for i in range(len(a)):
            if a[i][0][1][0] != b[i][1].numpy()[0][0]:
                assert False


def test_multi_process_data_loader_new():
    """
    test multi process data loader
    :return:
    """
    with fluid.dygraph.guard():
        train_reader = paddle.batch(
                    reader_decorator(
                        paddle.dataset.mnist.train()),
                        batch_size=1,
                        drop_last=True)
        train_loader = fluid.io.DataLoader.from_generator(capacity=2, use_multiprocess=True)
        train_loader.set_sample_list_generator(train_reader, places=fluid.CPUPlace())
        while True:
            a = next_sample(train_reader)
            b = next_sample(train_loader)
            if a == b and a is None:
                break
            if a[0][1][0] != b[1].numpy()[0][0]:
                assert False

if __name__ == '__main__':
    test_multi_process_data_loader()