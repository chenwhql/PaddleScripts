import contextlib
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.dygraph.base import to_variable
import six

@contextlib.contextmanager
def new_program_scope(main=None, startup=None, scope=None):
    prog = main if main else fluid.Program()
    startup_prog = startup if startup else fluid.Program()
    scope = scope if scope else fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            with fluid.unique_name.guard():
                yield

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
      
hidden_size = 10
vocab_size = 1000
num_steps = 3
init_scale = 0.1
batch_size = 4
batch_num = 200

for is_sparse in [True]:#[False, True]:
    # dygraph
    with fluid.dygraph.guard():
        # why need program
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1

        simple_net = SimpleNet(
            "simple_net",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_steps=num_steps,
            init_scale=init_scale,
            is_sparse=is_sparse)
        
        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        dy_param_updated = dict()
        dy_param_init = dict()
        dy_loss = None

        for i in range(1):#(batch_num):
            x_data = np.arange(12).reshape(4, 3).astype('int64')
            y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
            x_data = x_data.reshape((-1, num_steps, 1))
            y_data = y_data.reshape((-1, 1))

            x = to_variable(x_data)
            y = to_variable(y_data)
            outs = simple_net(x, y)

            dy_loss = outs
            if i == 0:
                for param in simple_net.parameters():
                    dy_param_init[param.name] = param.numpy()
            dy_loss.backward()
            sgd.minimize(dy_loss)
            simple_net.clear_gradients()
            if i == batch_num - 1:
                for param in simple_net.parameters():
                    dy_param_updated[param.name] = param.numpy()
        dy_loss_value = dy_loss.numpy()
    """
    # static
    with new_program_scope():
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1

        simple_net = SimpleNet(
            "simple_net",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_steps=num_steps,
            is_sparse=is_sparse)
        
        exe = fluid.Executor(fluid.CPUPlace()
                            if not fluid.core.is_compiled_with_cuda()
                            else fluid.CUDAPlace(0))

        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        x = fluid.layers.data(
            name="x", shape=[-1, num_steps, 1], dtype='int64')
        y = fluid.layers.data(
            name="y", shape=[-1, 1], dtype='float32')

        static_loss = simple_net(x, y)
        sgd.minimize(static_loss)

        static_param_updated = dict()
        static_param_init = dict()
        static_param_name_list = list()
        for param in simple_net.parameters():
            static_param_name_list.append(param.name)

        out = exe.run(fluid.default_startup_program(),
                      fetch_list=static_param_name_list)

        for i in range(len(static_param_name_list)):
            static_param_init[static_param_name_list[i]] = out[i]

        static_loss_value = None
        for i in range(batch_num):
            x_data = np.arange(12).reshape(4, 3).astype('int64')
            y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
            x_data = x_data.reshape((-1, num_steps, 1))
            y_data = y_data.reshape((-1, 1))
            fetch_list = [static_loss]
            fetch_list.extend(static_param_name_list)
            out = exe.run(fluid.default_main_program(),
                          feed={"x": x_data,
                                "y": y_data},
                          fetch_list=fetch_list)
            static_loss_value = out[0]

            if i == batch_num - 1:
                for k in range(3, len(out)):
                    static_param_updated[static_param_name_list[
                        k - 1]] = out[k]
    """
    print("Sparse mode: " if is_sparse else "Dense mode:")
    #print("- static graph loss: %.6f" % static_loss_value[0])
    print("- dygrah loss: %.6f" % dy_loss_value[0])
    #if not np.array_equal(static_loss_value, dy_loss_value):
    #    print(static_loss_value)
    #    print(dy_loss_value)   
    """
    for key, value in six.iteritems(static_param_init):
        if not np.array_equal(value, dy_param_init[key]):
            print(value)
            print(dy_param_init[key])
    for key, value in six.iteritems(static_param_updated):
        if not np.array_equal(value, dy_param_updated[key]):
            print(value)
            print(dy_param_updated[key])
    """