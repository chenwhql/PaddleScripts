#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
import six
import tarfile

import paddle
import paddle.dataset.common
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.base import to_variable

# from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase

class SimpleLSTMRNN(fluid.Layer):
    def __init__(self,
                 name_scope,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__(name_scope)
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])
            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m

                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(fluid.Layer):
    def __init__(self,
                 name_scope,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False,
                 dropout=None):
        super(PtbModel, self).__init__(name_scope)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            self.full_name(),
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = Embedding(
            self.full_name(),
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def build_once(self, input, label, init_hidden, init_cell):
        pass

    def forward(self, input, label, init_hidden, init_cell):

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)

        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)

        rnn_out = fluid.layers.reshape(
            rnn_out, shape=[-1, self.num_steps, self.hidden_size])
        projection = fluid.layers.matmul(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        loss.permissions = True

        return loss, last_hidden, last_cell


EOS = "</eos>"


def build_vocab(filename):
    vocab_dict = {}
    ids = 0
    vocab_dict[EOS] = ids
    ids += 1

    for line in filename:
        for w in line.strip().split():
            if w not in vocab_dict:
                vocab_dict[w] = ids
                ids += 1

    return vocab_dict


def file_to_ids(src_file, src_vocab):
    src_data = []
    for line in src_file:
        arra = line.strip().split()
        ids = [src_vocab[w] for w in arra if w in src_vocab]

        src_data += ids + [0]
    return src_data


def ptb_train_reader():
    with tarfile.open(
                paddle.dataset.common.download(
                    paddle.dataset.imikolov.URL, 'imikolov',
                    paddle.dataset.imikolov.MD5)) as tf:
        train_file = tf.extractfile('./simple-examples/data/ptb.train.txt')

        vocab_dict = build_vocab(train_file)
        train_file.seek(0)
        train_ids = file_to_ids(train_file, vocab_dict)

    def __reader__():
        data_len = len(train_ids)
        raw_data = np.asarray(train_ids, dtype="int64")

        batch_len = data_len // batch_size

        data = raw_data[0:batch_size * batch_len].reshape((batch_size, batch_len))

        epoch_size = (batch_len - 1) // num_steps
        for i in range(epoch_size):
            x = np.copy(data[:, i * num_steps:(i + 1) * num_steps])
            y = np.copy(data[:, i * num_steps + 1:(i + 1) * num_steps + 1])

            yield (x, y)

    def __simple_reader__():
        raw_data = np.asarray(train_ids, dtype="int64")
        epoch_size = batch_size * batch_num
        for i in range(epoch_size):
            x = np.copy(raw_data[i * num_steps:(i + 1) * num_steps])
            y = np.copy(raw_data[i * num_steps + 1:(i + 1) * num_steps + 1])

            yield (x, y)

    return __simple_reader__


vocab_size = 10000
num_layers = 1
batch_size = 2
hidden_size = 10
num_steps = 3
init_scale = 0.1
dropout = 0.0
batch_num = 200

place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
with fluid.dygraph.guard(place):
    strategy = fluid.dygraph.parallel.prepare_context()
    ptb_model = PtbModel(
            "ptb_model",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_steps=num_steps,
            init_scale=init_scale,
            dropout=dropout,
            is_sparse=True)
    ptb_model = fluid.dygraph.parallel.DataParallel(ptb_model, strategy)

    train_reader = paddle.batch(
        ptb_train_reader(), batch_size=batch_size, drop_last=True)
    train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)

    sgd = SGDOptimizer(learning_rate=1e-3)

    dy_param_updated = dict()
    dy_param_init = dict()
    dy_loss = None
    last_hidden = None
    last_cell = None

    init_hidden_data = np.zeros(
        (num_layers, batch_size, hidden_size), dtype='float32')
    init_cell_data = np.zeros(
        (num_layers, batch_size, hidden_size), dtype='float32')

    for batch_id, batch in enumerate(train_reader()):
        x_data = np.array([x[0].reshape(num_steps) for x in batch]).astype('int64')
        y_data = np.array([x[1].reshape(num_steps) for x in batch]).astype('int64')
        x_data = x_data.reshape((-1, num_steps, 1))
        y_data = y_data.reshape((-1, 1))
        x = to_variable(x_data)
        y = to_variable(y_data)
        init_hidden = to_variable(init_hidden_data)
        init_cell = to_variable(init_cell_data)
        dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                    init_cell)

        out_loss = dy_loss.numpy()

        init_hidden_data = last_hidden.numpy()
        init_cell_data = last_cell.numpy()

        dy_loss = ptb_model.scale_loss(dy_loss)   
        dy_loss.backward()
        ptb_model.apply_collective_grads()

        sgd.minimize(dy_loss)
        ptb_model.clear_gradients()

    print("out loss: %.6f" % out_loss[0])