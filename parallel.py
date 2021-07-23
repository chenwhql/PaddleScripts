import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.base import to_variable

place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
with fluid.dygraph.guard(place=place):

    # prepare the data parallel context
    strategy=dygraph.prepare_context()

    linear = Linear(1, 10, act="softmax")
    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=linear.parameters())

    # make the module become the data parallelism module
    linear = dygraph.DataParallel(linear, strategy)

    x_data = np.random.random(size=[10, 1]).astype(np.float32)
    data = to_variable(x_data)

    hidden = linear(data)
    avg_loss = fluid.layers.mean(hidden)
    print(avg_loss)

    # scale the loss according to the number of trainers.
    # avg_loss = linear.scale_loss(avg_loss)

    # avg_loss.backward()

    # # collect the gradients of trainers.
    # linear.apply_collective_grads()

    # adam.minimize(avg_loss)
    # linear.clear_gradients()