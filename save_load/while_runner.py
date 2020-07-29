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
        default="./control_flow.while",
        help="Model save dir")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args

def train_while(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        # fluid.dygraph.StaticModelRunner(args.model_dir)
        model = fluid.dygraph.StaticModelRunner(args.model_dir)

        d = []
        for _ in range(3):
            d.append(np.random.random(size=[10]).astype('float32'))

        loss = model(inputs=[d[0], d[1], d[2]])
        
        avg_loss = fluid.layers.mean(loss)

        avg_loss.backward()

if __name__ == '__main__':
    args = parse_args()
    train_while(args)