#!/usr/bin/env python
# coding=utf-8
#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import numpy
import paddle
import paddle.fluid as fluid

def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def multilayer_perceptron(img, label):
    img = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    return loss_net(hidden, label)


def softmax_regression(img, label):
    return loss_net(img, label)


def convolutional_neural_network(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(nn_type,
          use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
    startup_program.random_seed = 90
    main_program.random_seed = 90

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if nn_type == 'softmax_regression':
        net_conf = softmax_regression
    elif nn_type == 'multilayer_perceptron':
        net_conf = multilayer_perceptron
    else:
        net_conf = convolutional_neural_network

    prediction, avg_loss, acc = net_conf(img, label)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(startup_program)

    for step_id, data in enumerate(train_reader()):
        metrics = exe.run(
            main_program,
            feed=feeder.feed(data),
            fetch_list=[avg_loss, acc])
        print("Cost %f" % (metrics[0]))
        break

def main(use_cuda, nn_type):
    train(
        nn_type=nn_type,
        use_cuda=use_cuda)

if __name__ == '__main__':
    BATCH_SIZE = 64
    use_cuda = False
    # predict = 'softmax_regression' # uncomment for Softmax
    # predict = 'multilayer_perceptron' # uncomment for MLP
    predict = 'convolutional_neural_network'  # uncomment for LeNet5
    main(use_cuda=use_cuda, nn_type=predict)

