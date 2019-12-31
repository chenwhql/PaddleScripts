from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


class GoogLeNet():
    def __init__(self):

        pass

    def conv_layer(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None,
                   name=None):
        channels = input.shape[1]
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=False,
            name=name)
        return conv

    def xavier(self, channels, filter_size, name):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")

        return param_attr

    def inception(self,
                  input,
                  channels,
                  filter1,
                  filter3R,
                  filter3,
                  filter5R,
                  filter5,
                  proj,
                  name=None):
        conv1 = self.conv_layer(
            input=input,
            num_filters=filter1,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_1x1")
        conv3r = self.conv_layer(
            input=input,
            num_filters=filter3R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3_reduce")
        conv3 = self.conv_layer(
            input=conv3r,
            num_filters=filter3,
            filter_size=3,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3")
        conv5r = self.conv_layer(
            input=input,
            num_filters=filter5R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5_reduce")
        conv5 = self.conv_layer(
            input=conv5r,
            num_filters=filter5,
            filter_size=5,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5")
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=1,
            pool_padding=1,
            pool_type='max')
        convprj = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=proj,
            stride=1,
            padding=0,
            name="inception_" + name + "_3x3_proj",
            param_attr=ParamAttr(
                name="inception_" + name + "_3x3_proj_weights"),
            bias_attr=False)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat

    def net(self, input, class_dim=1000):
        conv = self.conv_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act=None,
            name="conv1")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        conv = self.conv_layer(
            input=pool,
            num_filters=64,
            filter_size=1,
            stride=1,
            act=None,
            name="conv2_1x1")
        conv = self.conv_layer(
            input=conv,
            num_filters=192,
            filter_size=3,
            stride=1,
            act=None,
            name="conv2_3x3")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        ince3a = self.inception(pool, 192, 64, 96, 128, 16, 32, 32, "ince3a")
        ince3b = self.inception(ince3a, 256, 128, 128, 192, 32, 96, 64,
                                "ince3b")
        pool3 = fluid.layers.pool2d(
            input=ince3b, pool_size=3, pool_type='max', pool_stride=2)

        ince4a = self.inception(pool3, 480, 192, 96, 208, 16, 48, 64, "ince4a")
        ince4b = self.inception(ince4a, 512, 160, 112, 224, 24, 64, 64,
                                "ince4b")
        ince4c = self.inception(ince4b, 512, 128, 128, 256, 24, 64, 64,
                                "ince4c")
        ince4d = self.inception(ince4c, 512, 112, 144, 288, 32, 64, 64,
                                "ince4d")
        ince4e = self.inception(ince4d, 528, 256, 160, 320, 32, 128, 128,
                                "ince4e")
        pool4 = fluid.layers.pool2d(
            input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = self.inception(pool4, 832, 256, 160, 320, 32, 128, 128,
                                "ince5a")
        ince5b = self.inception(ince5a, 832, 384, 192, 384, 48, 128, 128,
                                "ince5b")
        pool5 = fluid.layers.pool2d(
            input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout,
                              size=class_dim,
                              act='softmax',
                              param_attr=self.xavier(1024, 1, "out"),
                              name="out",
                              bias_attr=ParamAttr(name="out_offset"))

        pool_o1 = fluid.layers.pool2d(
            input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = self.conv_layer(
            input=pool_o1,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o1")
        fc_o1 = fluid.layers.fc(input=conv_o1,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o1"),
                                name="fc_o1",
                                bias_attr=ParamAttr(name="fc_o1_offset"))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out1"),
                               name="out1",
                               bias_attr=ParamAttr(name="out1_offset"))

        pool_o2 = fluid.layers.pool2d(
            input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = self.conv_layer(
            input=pool_o2,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o2")
        fc_o2 = fluid.layers.fc(input=conv_o2,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o2"),
                                name="fc_o2",
                                bias_attr=ParamAttr(name="fc_o2_offset"))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out2"),
                               name="out2",
                               bias_attr=ParamAttr(name="out2_offset"))

        # last fc layer is "out"
        return out, out1, out2

def reader_decorator(reader):
    def _reader_imple():
        for item in reader():
            doc = np.array(item[0]).reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield doc, label

    return _reader_imple

def train_test():
    batch_size = 32

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    img = fluid.data(name='img', shape=[None, 3, 224, 224], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    
    model = GoogLeNet()
    out, out1, out2 = model.net(input=img)
    cost0 = fluid.layers.cross_entropy(input=out, label=label)
    cost1 = fluid.layers.cross_entropy(input=out1, label=label)
    cost2 = fluid.layers.cross_entropy(input=out2, label=label)
    avg_cost0 = fluid.layers.mean(x=cost0)
    avg_cost1 = fluid.layers.mean(x=cost1)
    avg_cost2 = fluid.layers.mean(x=cost2)

    avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)

    #test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_cost)
    
    main_program = fluid.default_main_program()
    test_program = fluid.default_main_program().clone(for_test=True)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    train_reader = paddle.batch(
        reader_decorator(
            paddle.dataset.flowers.train(use_xmap=False)),
        batch_size=batch_size,
        drop_last=True)
    test_reader = paddle.batch(
        reader_decorator(
            paddle.dataset.flowers.test(use_xmap=False)),
        batch_size=batch_size,
        drop_last=True)
    
    exe.run(fluid.default_startup_program())

    for step_id, data in enumerate(train_reader()):
        loss = exe.run(main_program,
                       feed=feeder.feed(data),
                       fetch_list=[avg_cost, acc_top1])

        if step_id % 10 == 0:
            print("train step: %d, loss: %.3f" % (step_id, loss[0]))
            break

    for step_id, data in enumerate(test_reader()):
        loss = exe.run(test_program,
                       feed=feeder.feed(data),
                       fetch_list=[avg_cost, acc_top1])
        if step_id % 10 == 0:
            print("test step: %d, loss: %.3f" % (step_id, loss[0]))
            break
    
if __name__ == '__main__':
    train_test()
    

    