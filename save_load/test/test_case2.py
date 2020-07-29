import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import declarative
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

'''
Global Configs
'''
USE_CUDA = True
EPOCH_NUM = 2
BATCH_SIZE = 64
MODEL_PATH = "mnist.imperative.to.declarative"

'''
Part 1. Model
'''
class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    @declarative
    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        return x

'''
Part 2. Assist Functions
'''
def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(1, 28, 28)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__

def train_one_epoch(layer, train_loader):
    for batch_id, data in enumerate(train_loader()):
        img, label = data
        label.stop_gradient = True

        cost = layer(img)
        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)

        avg_loss.backward()
        adam.minimize(avg_loss)
        layer.clear_gradients()

        if batch_id % 200 == 0:
            print("Loss at step {}: {:}".format(
                batch_id, avg_loss.numpy()))
    return avg_loss

'''
Part 3. Train & Save
'''
# enable dygraph mode
place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
fluid.enable_dygraph(place) 
# create network
mnist = MNIST()
adam = AdamOptimizer(
    learning_rate=0.001, parameter_list=mnist.parameters())
# create train data loader
train_reader = paddle.batch(
    reader_decorator(paddle.dataset.mnist.train()),
    batch_size=BATCH_SIZE,
    drop_last=True)
train_loader = fluid.io.DataLoader.from_generator(capacity=5)
train_loader.set_sample_list_generator(train_reader, places=place)
# train
for epoch in range(EPOCH_NUM):
    train_one_epoch(mnist, train_loader)
# save
fluid.dygraph.jit.save(layer=mnist, model_path=MODEL_PATH)

'''
Part 4. Load & Inference
'''
# load model by jit.load & inference
translated_mnist = fluid.dygraph.jit.load(model_path=MODEL_PATH)
translated_mnist.eval()
image = np.random.random((1, 1, 28, 28)).astype('float32')
image_var = fluid.dygraph.to_variable(image)
dygraph_pred = translated_mnist(image_var)
# load model by io.load_inference_model & inference
fluid.disable_dygraph()
exe = fluid.Executor(place)
[infer_program, feed, fetch] = fluid.io.load_inference_model(dirname=MODEL_PATH, executor=exe, params_filename="__variables__")
static_pred = exe.run(infer_program,
    feed={feed[0]: image},
    fetch_list=fetch)
# compare
print("dygraph prediction: {}".format(dygraph_pred.numpy()))
print("static prediction: {}".format(static_pred[0]))
np.testing.assert_array_equal(dygraph_pred.numpy(), static_pred[0])
