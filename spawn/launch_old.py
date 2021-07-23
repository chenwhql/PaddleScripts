import numpy as np
import paddle.fluid as fluid

place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
with fluid.dygraph.guard(place):

    # prepare the data parallel context
    strategy = fluid.dygraph.prepare_context()

    linear = fluid.dygraph.Linear(1, 10, act="softmax")
    adam = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, parameter_list=linear.parameters())

    # make the module become the data parallelism module
    linear = fluid.dygraph.DataParallel(linear, strategy)

    x_data = np.random.random(size=[10, 1]).astype(np.float32)
    data = fluid.dygraph.to_variable(x_data)

    hidden = linear(data)
    avg_loss = fluid.layers.mean(hidden)

    # scale the loss according to the number of trainers.
    avg_loss = linear.scale_loss(avg_loss)

    avg_loss.backward()

    # collect the gradients of trainers.
    linear.apply_collective_grads()

    adam.minimize(avg_loss)
    linear.clear_gradients()