# Include libraries.
import os
import sys

import numpy

import paddle
import paddle.fluid as fluid


# Configure the neural network.
def net(x, y):
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)
    return avg_loss


def fake_reader():
    def reader():
        for i in range(1000):
            x = numpy.random.random((1, 13)).astype('float32')
            y = numpy.random.randint(0, 2, (1, 1)).astype('float32')
            yield x,y
    return reader


# Define train function.
def train():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    label = fluid.layers.data(name='y', shape=[1], dtype='float32')
    avg_cost = net(x, label)

    sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    train_reader = paddle.batch(fake_reader(), batch_size=4)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program, is_chief=False):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
        exe.run(fluid.default_startup_program())

        PASS_NUM = 2
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                avg_loss_value, = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                print("pass %d, total avg cost = %f" % (pass_id, avg_loss_value))

    train_loop(fluid.default_main_program())

# Run train and infer.
if __name__ == '__main__':
    train()

