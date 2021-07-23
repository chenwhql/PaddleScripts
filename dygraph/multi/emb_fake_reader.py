import numpy as np
import paddle
import paddle.dataset.common
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.dygraph.base import to_variable
import six
import tarfile

class SimpleNet(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False):
        super(SimpleNet, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        self.embedding = Embedding(
            size=[self.vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_param',
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

    def forward(self, input, label):
        x_emb = self.embedding(input)
        fc = fluid.layers.matmul(x_emb, self.softmax_weight)
        fc = fluid.layers.elementwise_add(fc, self.softmax_bias)
        projection = fluid.layers.reshape(
            fc, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        return loss

# global configs
batch_size = 4
batch_num = 200
hidden_size = 10
vocab_size = 1000
num_steps = 3
init_scale = 0.1


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.arange(num_steps).astype('int64')
            y_data = np.arange(1, 1 + num_steps).astype('int64')
            yield x_data, y_data

    return __reader__


place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
with fluid.dygraph.guard(place):
    strategy = fluid.dygraph.parallel.prepare_context()
    simple_net = SimpleNet(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_steps=num_steps,
        init_scale=init_scale,
        is_sparse=True)
    simple_net = fluid.dygraph.parallel.DataParallel(simple_net, strategy)

    train_reader = paddle.batch(
        fake_sample_reader(), batch_size=batch_size, drop_last=True)
    train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)
    
    sgd = fluid.optimizer.SGD(learning_rate=1e-3, parameter_list=simple_net.parameters())
    dy_loss = None

    for i, data in enumerate(train_reader()):
        x_data = np.array([x[0].reshape(3) for x in data]).astype('int64')
        y_data = np.array([x[1].reshape(3) for x in data]).astype('int64')
        x_data = x_data.reshape((-1, num_steps, 1))
        y_data = y_data.reshape((-1, 1))

        x = to_variable(x_data)
        y = to_variable(y_data)
        dy_loss = simple_net(x, y)

        dy_loss = simple_net.scale_loss(dy_loss)   
        dy_loss.backward()
        simple_net.apply_collective_grads()

        sgd.minimize(dy_loss)
        simple_net.clear_gradients()
    dy_loss_value = dy_loss.numpy()

print("- dygrah loss: %.6f" % dy_loss_value[0])
