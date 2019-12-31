#coding:utf-8

import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import numpy
import os

use_pyReader = True

def my_reader():
    def reader():
        for i in range(160):
            train_data = numpy.random.randint(0, 10, [50]).astype('int64')
            label_data = numpy.random.randint(0, 2, size=(1)).astype('int64')
            yield train_data, label_data
    return reader

train_reader = fluid.io.batch(paddle.reader.shuffle(my_reader(),
                                        buf_size=8), 4)

DEV_COUNT = fluid.core.get_cuda_device_count()
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

if not use_cuda:
    os.environ['CPU_NUM'] = str(2)

data = fluid.data(name="char", shape=[None, 50], dtype="int64", lod_level=0)
label = fluid.data(name="label", shape=[None, 1], dtype="int64", lod_level=0)

if use_pyReader:
  reader = fluid.io.PyReader(feed_list=[data, label], capacity=40, iterable=True, return_list=False)
  reader.decorate_sample_list_generator(train_reader, place)

emb = fluid.embedding(data, size=[10, 64])
prob = fluid.layers.fc(emb, size=2, act='softmax')
ce = fluid.layers.cross_entropy(prob, label)
loss = fluid.layers.mean(ce)

exe = fluid.Executor(place)
fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(fluid.default_startup_program())
build_strategy = fluid.BuildStrategy()
build_strategy.fuse_all_reduce_ops = False
compiled_train_prog = compiler.CompiledProgram(
         fluid.default_main_program()).with_data_parallel(
                  loss_name=loss.name, build_strategy=build_strategy)

feed_data = []
if use_pyReader:
    for data in reader():
        if len(feed_data) < DEV_COUNT:
            feed_data.extend(data)
        else: 
            break
else:
    train_data = numpy.random.randint(0,10, (4, 50)).astype('int64')
    label_data = numpy.random.randint(0,2, size=(4, 1)).astype('int64')
    for i in range(DEV_COUNT):
        feed_data.extend([{"char": train_data, "label": label_data}])

loss_data = exe.run(compiled_train_prog,
                    feed=feed_data,
                    fetch_list=[loss.name])

print(loss_data)