# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import numpy as np
import paddle
import paddle.fluid as fluid

import paddle.fluid.transpiler.details.program_utils as pu

def parse_args():
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help="Whether to use GPU or not.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./recognize_digits_convolutional_neural_network.inference.model",
        help="Model save dir")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args

def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(1, 28, 28)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label
    return __reader__

def train_mnist(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    BATCH_SIZE = 64

    with fluid.dygraph.guard(place):
        mnist = fluid.dygraph.StaticModelRunner(args.model_dir)

        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())

        train_reader = paddle.batch(
            reader_decorator(
                paddle.dataset.mnist.train()), 
                batch_size=BATCH_SIZE,
                drop_last=True)
        train_loader = fluid.io.DataLoader.from_generator(capacity=10)
        train_loader.set_sample_list_generator(train_reader, places=place)

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            # mnist(inputs=img)
            cost = mnist(inputs=img)
            
            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = fluid.layers.mean(loss)

            avg_loss.backward()

            adam.minimize(avg_loss)
            
            break


if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)