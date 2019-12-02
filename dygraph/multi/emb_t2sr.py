import numpy as np
import paddle
import paddle.dataset.common
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.dygraph.base import to_variable
import six
import tarfile

def fake_reader():
    def __reader__():
        iteration = batch_size * batch_num
        for _ in six.moves.range(iteration):
            x = np.arange(12)
            y = np.arange(1, 13)
            yield x, y
    return __reader__()

class SimpleNet(fluid.Layer):
    def __init__(self,
                 name_scope,
                 hidden_size,
                 vocab_size,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False):
        super(SimpleNet, self).__init__(name_scope)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        self.embedding = Embedding(
            self.full_name(),
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_param',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype='float32',
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label):
        x_emb = self.embedding(input)
        projection = fluid.layers.matmul(
            x_emb, fluid.layers.transpose(
                self.embedding._w, perm=[1,0]))
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        loss.permissions = True

        return loss
      
batch_size = 4
batch_num = 200
hidden_size = 10
vocab_size = 1000
num_steps = 3
init_scale = 0.1
batch_num = 200

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
        raw_data = np.asarray(train_ids, dtype="int64")
        epoch_size = batch_size * batch_num
        for i in range(epoch_size):
            x = np.copy(raw_data[i * num_steps:(i + 1) * num_steps])
            y = np.copy(raw_data[i * num_steps + 1:(i + 1) * num_steps + 1])

            yield (x, y)

    return __reader__

place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
with fluid.dygraph.guard(place):
    strategy = fluid.dygraph.parallel.prepare_context()
    simple_net = SimpleNet(
        "simple_net",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_steps=num_steps,
        init_scale=init_scale,
        is_sparse=True)
    simple_net = fluid.dygraph.parallel.DataParallel(simple_net, strategy)

    train_reader = paddle.batch(
        ptb_train_reader(), batch_size=batch_size, drop_last=True)
    train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)
    
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
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

# print("Sparse mode: " if is_sparse else "Dense mode:")
print("- dygrah loss: %.6f" % dy_loss_value[0])
